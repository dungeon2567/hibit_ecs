#include "ecs.h"
#include <ctype.h>
#include <string.h>
#include <stddef.h>

/* String → ecs_compiled_query_t.

   Grammar (recursive descent):
     expr  := or
     or    := and ('|' and)*
     and   := unary ('&' unary)*
     unary := '!' unary | atom
     atom  := IDENT | '(' or ')'

   Identifiers map to trees by name lookup in world->mask. AST → DNF: each
   disjunct becomes one clause with include = positives, exclude = negatives. */

#define ECS_QUERY_MAX_NODES 64
#define ECS_QUERY_MAX_DNF   64

typedef enum {
    TOK_END, TOK_IDENT, TOK_AND, TOK_OR, TOK_NOT, TOK_LPAREN, TOK_RPAREN, TOK_ERR
} tok_kind_t;

typedef struct {
    const char* p;
    tok_kind_t  kind;
    const char* tok_start;
    int         tok_len;
} lex_t;

typedef enum { N_ATOM, N_NOT, N_AND, N_OR } node_kind_t;

typedef struct node {
    node_kind_t  kind;
    int          atom_idx;
    struct node* a;
    struct node* b;
} node_t;

typedef struct {
    lex_t       L;
    node_t      nodes[ECS_QUERY_MAX_NODES];
    int         n_count;
    const char* atom_name[ECS_QUERY_MAX_TERMS];
    int         atom_len[ECS_QUERY_MAX_TERMS];
    int         atom_count;
    int         err;
} parser_t;

typedef struct {
    uint32_t pos;
    uint32_t neg;
} dnf_clause_t;

typedef struct {
    dnf_clause_t cl[ECS_QUERY_MAX_DNF];
    int          count;
} dnf_t;

static void lex_advance(lex_t* L) {
    while (*L->p && isspace((unsigned char)*L->p)) L->p++;
    L->tok_start = L->p;
    L->tok_len   = 0;
    char c = *L->p;
    if (!c)        { L->kind = TOK_END;    return; }
    if (c == '&')  { L->p++; L->kind = TOK_AND;    return; }
    if (c == '|')  { L->p++; L->kind = TOK_OR;     return; }
    if (c == '!')  { L->p++; L->kind = TOK_NOT;    return; }
    if (c == '(')  { L->p++; L->kind = TOK_LPAREN; return; }
    if (c == ')')  { L->p++; L->kind = TOK_RPAREN; return; }
    if (isalpha((unsigned char)c) || c == '_') {
        const char* s = L->p;
        while (isalnum((unsigned char)*L->p) || *L->p == '_') L->p++;
        L->tok_start = s;
        L->tok_len   = (int)(L->p - s);
        L->kind      = TOK_IDENT;
        return;
    }
    L->kind = TOK_ERR;
}

static node_t* alloc_node(parser_t* P) {
    if (P->n_count >= ECS_QUERY_MAX_NODES) { P->err = 1; return NULL; }
    return &P->nodes[P->n_count++];
}

static int intern_atom(parser_t* P, const char* s, int n) {
    for (int i = 0; i < P->atom_count; i++)
        if (P->atom_len[i] == n && memcmp(P->atom_name[i], s, (size_t)n) == 0)
            return i;
    if (P->atom_count >= ECS_QUERY_MAX_TERMS) { P->err = 1; return -1; }
    P->atom_name[P->atom_count] = s;
    P->atom_len[P->atom_count]  = n;
    return P->atom_count++;
}

static node_t* parse_or(parser_t* P);

static node_t* parse_atom(parser_t* P) {
    if (P->L.kind == TOK_LPAREN) {
        lex_advance(&P->L);
        node_t* n = parse_or(P);
        if (!n) return NULL;
        if (P->L.kind != TOK_RPAREN) { P->err = 1; return NULL; }
        lex_advance(&P->L);
        return n;
    }
    if (P->L.kind == TOK_IDENT) {
        int idx = intern_atom(P, P->L.tok_start, P->L.tok_len);
        lex_advance(&P->L);
        if (idx < 0) return NULL;
        node_t* n = alloc_node(P);
        if (!n) return NULL;
        n->kind = N_ATOM; n->atom_idx = idx; n->a = NULL; n->b = NULL;
        return n;
    }
    P->err = 1;
    return NULL;
}

static node_t* parse_unary(parser_t* P) {
    if (P->L.kind == TOK_NOT) {
        lex_advance(&P->L);
        node_t* c = parse_unary(P);
        if (!c) return NULL;
        node_t* n = alloc_node(P);
        if (!n) return NULL;
        n->kind = N_NOT; n->atom_idx = 0; n->a = c; n->b = NULL;
        return n;
    }
    return parse_atom(P);
}

static node_t* parse_and(parser_t* P) {
    node_t* lhs = parse_unary(P);
    if (!lhs) return NULL;
    while (P->L.kind == TOK_AND) {
        lex_advance(&P->L);
        node_t* rhs = parse_unary(P);
        if (!rhs) return NULL;
        node_t* n = alloc_node(P);
        if (!n) return NULL;
        n->kind = N_AND; n->atom_idx = 0; n->a = lhs; n->b = rhs;
        lhs = n;
    }
    return lhs;
}

static node_t* parse_or(parser_t* P) {
    node_t* lhs = parse_and(P);
    if (!lhs) return NULL;
    while (P->L.kind == TOK_OR) {
        lex_advance(&P->L);
        node_t* rhs = parse_and(P);
        if (!rhs) return NULL;
        node_t* n = alloc_node(P);
        if (!n) return NULL;
        n->kind = N_OR; n->atom_idx = 0; n->a = lhs; n->b = rhs;
        lhs = n;
    }
    return lhs;
}

static int dnf_and(const dnf_t* A, const dnf_t* B, dnf_t* out) {
    out->count = 0;
    for (int i = 0; i < A->count; i++) {
        for (int j = 0; j < B->count; j++) {
            uint32_t pos = A->cl[i].pos | B->cl[j].pos;
            uint32_t neg = A->cl[i].neg | B->cl[j].neg;
            if (pos & neg) continue;
            if (out->count >= ECS_QUERY_MAX_DNF) return 0;
            out->cl[out->count].pos = pos;
            out->cl[out->count].neg = neg;
            out->count++;
        }
    }
    return 1;
}

static int dnf_or(const dnf_t* A, const dnf_t* B, dnf_t* out) {
    if (A->count + B->count > ECS_QUERY_MAX_DNF) return 0;
    out->count = 0;
    for (int i = 0; i < A->count; i++) out->cl[out->count++] = A->cl[i];
    for (int j = 0; j < B->count; j++) out->cl[out->count++] = B->cl[j];
    return 1;
}

static int dnf_build(node_t* n, int neg, dnf_t* out) {
    switch (n->kind) {
    case N_ATOM:
        out->count = 1;
        out->cl[0].pos = neg ? 0u : (1u << n->atom_idx);
        out->cl[0].neg = neg ? (1u << n->atom_idx) : 0u;
        return 1;
    case N_NOT:
        return dnf_build(n->a, !neg, out);
    case N_AND: {
        dnf_t A = {0}, B = {0};
        if (!dnf_build(n->a, neg, &A)) return 0;
        if (!dnf_build(n->b, neg, &B)) return 0;
        /* neg ? !(a&b) = !a | !b : a & b */
        return neg ? dnf_or(&A, &B, out) : dnf_and(&A, &B, out);
    }
    case N_OR: {
        dnf_t A = {0}, B = {0};
        if (!dnf_build(n->a, neg, &A)) return 0;
        if (!dnf_build(n->b, neg, &B)) return 0;
        /* neg ? !(a|b) = !a & !b : a | b */
        return neg ? dnf_and(&A, &B, out) : dnf_or(&A, &B, out);
    }
    }
    return 0;
}

ecs_compiled_query_t* ecs_compile_query(const ecs_world_t* world, const char* expr) {
    if (!world || !expr) return NULL;

    parser_t P = {0};
    P.L.p = expr;
    lex_advance(&P.L);

    node_t* root = parse_or(&P);
    if (!root || P.err || P.L.kind != TOK_END) return NULL;

    dnf_t d = {0};
    if (!dnf_build(root, 0, &d)) return NULL;
    if (d.count == 0 || d.count > ECS_QUERY_MAX_CLAUSES) return NULL;
    if (P.atom_count > ECS_QUERY_MAX_TERMS) return NULL;

    /* Resolve atom name → tree pointer via world->mask. */
    ecs_tree_t* resolved[ECS_QUERY_MAX_TERMS] = {0};
    for (int a = 0; a < P.atom_count; a++) {
        ecs_tree_t* found = NULL;
        uint64_t mask = world->mask;
        while (mask) {
            int i = ecs_ctz64(mask); mask &= mask - 1;
            const char* nm = world->trees[i].name;
            if (nm
                && (int)strlen(nm) == P.atom_len[a]
                && memcmp(nm, P.atom_name[a], (size_t)P.atom_len[a]) == 0) {
                found = (ecs_tree_t*)&world->trees[i];
                break;
            }
        }
        if (!found) return NULL;
        resolved[a] = found;
    }

    ecs_compiled_query_t* out = (ecs_compiled_query_t*)ecs_xcalloc(1, sizeof(*out));
    out->world        = world;
    out->tree_count   = (uint32_t)P.atom_count;
    out->clause_count = (uint32_t)d.count;
    for (int a = 0; a < P.atom_count; a++) out->trees[a] = resolved[a];

    for (int c = 0; c < d.count; c++) {
        out->clauses[c].include = d.cl[c].pos;
        out->clauses[c].exclude = d.cl[c].neg;
    }
    return out;
}
