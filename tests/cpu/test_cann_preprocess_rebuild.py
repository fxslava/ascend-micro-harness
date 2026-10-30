"""Regression controls for CANN's in-place preprocessing link workaround."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_cann_preprocess import prepare


def elf(kind, endian=1):
    return b"\x7fELF" + bytes([2, endian]) + bytes(10) + kind.to_bytes(2, "little")


class PrepareTests(unittest.TestCase):
    def write(self, root, target, name, data):
        path = root / "CMakeFiles" / (target + ".dir") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_only_consumed_core_objects_are_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            consumed = [self.write(root, target, "source.cpp.o", elf(2))
                        for target in ("aic_obj", "aiv_obj", "m200_obj")]
            keep = [self.write(root, target, name, data) for target, name, data in (
                ("host_obj", "host.cpp.o", elf(2)),
                ("preprocess_obj", "preprocessed.cpp.o", b"preprocessed text"),
                ("aic_obj", "relocatable.cpp.o", elf(1)),
                ("aic_obj", "big_endian.cpp.o", elf(2, endian=2)),
                ("aiv_obj", "empty.cpp.o", b""),
                ("aiv_obj", "source.cpp", elf(2)),
            )]
            original = {path: path.read_bytes() for path in keep}
            self.assertEqual(prepare(root), 3)
            self.assertTrue(all(not path.exists() for path in consumed))
            self.assertEqual({path: path.read_bytes() for path in keep}, original)
            self.assertEqual(prepare(root), 0)

    def test_fresh_build_is_a_noop(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(prepare(folder), 0)

    def test_symlink_cannot_remove_an_outside_object(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root = base / "build"
            link = root / "CMakeFiles" / "aic_obj.dir" / "outside.o"
            link.parent.mkdir(parents=True)
            outside = base / "outside.o"
            outside.write_bytes(elf(2))
            link.symlink_to(outside)
            with self.assertRaises(ValueError):
                prepare(root)
            self.assertEqual(outside.read_bytes(), elf(2))
            self.assertTrue(link.is_symlink())


if __name__ == "__main__":
    unittest.main()
