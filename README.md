# Computer Animation — Columbia CS

Programming assignments from Columbia's Computer Animation course, built on the
course's `nemo` framework (Python, `uv`, ImGui viewport).

| | Assignment |
|---|---|
| **PA1–PA3** | Early assignments — see each folder's PDF for the brief |
| **PA4** | `nemo-pa4-release` |
| **PA5** | `nemo-pa5-release` |
| **PA6** | `nemo-pa6-release` |

Each `nemo-*` release has the same shape:

```
src/           implementation
assignments/   the tasks
scenes/        test scenes
tests/         verification
```

## Running

Each assignment is a standalone `uv` project:

```bash
cd nemo-pa6-release
uv sync
uv run pytest        # tests
```

---

Coursework, published for reference. © 2026 Chloe Joo-yeon Lee
