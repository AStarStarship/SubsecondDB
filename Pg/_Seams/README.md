# SubsecondDb — Postgres Extension (Pg)

SubsecondDb is a Postgres extension that exposes the SubsecondId archive tier as
SQL: the 64-bit SubsecondId as a native type, hot-to-archive compaction as a
function, and the collision ledger as a table. This `_Seams/` directory holds
the seam-test harness for the extension (build files + test scaffolding).

**Status: scaffold.** The build files (`CMakeFiles.txt`, VS project) and seam
harness are in place; the extension C source (`.c`/`.cpp` + `--sql` install
script) lands next. The GPU scheduler (`../Gpu/`) is the first fully-built,
seam-tested component.

See the root [README](../../README.md) for the full three-tier architecture
(hot KV + GPU vector + Postgres archive) and the SubsecondId inode design.

## License

Copyright [AStarship™](https://astarship.net).
