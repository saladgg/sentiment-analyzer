# Audit backlogs

Each numbered pair is one audit. `NN_<topic>.md` holds the findings, with
evidence and code references. `NN_<topic>_todos.md` is the execution
checklist; IDs match across the pair, and each todos file has a status
stamp at the top. Within a backlog, ship in tier order.

| Pair | Topic | Status stamp lives in |
| ---- | ----- | --------------------- |
| 01 | General audit (security, hardening, architecture, correctness, quality, performance, ops, testing, missing capabilities) | [`01_general_audit_todos.md`](01_general_audit_todos.md): **filed 2026-11-25, nothing shipped** |

Conventions: findings are severity-tagged (🔴/🟠/🟡/🟢), with an effort
estimate per item (**S** ≤ 0.5d · **M** ≤ 2d · **L** > 2d). Each finding is
marked ✅ Verified (reproduced) or 🔎 Code-evident (follows from the code,
not executed). Cross-backlog overlaps are declared in each file's header so
that one item never appears in two checklists.
