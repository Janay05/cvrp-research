# Legacy tests

`test_undo.cpp` is a standalone correctness test for the do/undo-list rollback
mechanism (`DoUndoEntry`/`ThreadArena`), written during earlier development of
the current solver. It predates several fields the real `ThreadArena` in
`src/ThreadArena.hpp` now has (e.g. `undoCostC`/`undoCostS`, see that file's
comments) and its own copies of `DoUndoEntry`/`ThreadArena` are stale — it will
not compile against the current `src/` as-is.

Kept because the underlying test logic (verifying insert/remove undo pairs
restore exact prior state) is still a reasonable regression test if updated to
match the current struct definitions. Not currently wired into any build or
CI — update it against `src/ThreadArena.hpp`'s current fields before relying
on it again.
