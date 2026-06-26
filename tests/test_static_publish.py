import tempfile
import unittest
from pathlib import Path

from tools import static_publish


class StaticPublishTest(unittest.TestCase):
    def test_static_page_paths_include_canonical_game_spec(self):
        self.assertIn(Path("game_spec.json"), static_publish.STATIC_PAGE_PATHS)

    def test_copy_static_pages_files_replaces_existing_directories(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            root = Path(src)
            worktree = Path(dst)
            (root / "js").mkdir()
            (root / "js" / "app.js").write_text("new", encoding="utf-8")
            (root / "index.html").write_text("<main></main>", encoding="utf-8")
            (root / "game_spec.json").write_text("{}", encoding="utf-8")

            (worktree / "js").mkdir()
            (worktree / "js" / "stale.js").write_text("old", encoding="utf-8")

            static_publish.copy_static_pages_files(worktree, root=root)

            self.assertEqual((worktree / "js" / "app.js").read_text(encoding="utf-8"), "new")
            self.assertFalse((worktree / "js" / "stale.js").exists())
            self.assertTrue((worktree / "game_spec.json").exists())


if __name__ == "__main__":
    unittest.main()
