import os

import pytest
from ecl.util.test import TestAreaContext
from utils import ResTest, tmpdir

from res.enkf import EnkfFs, EnKFMain
from res.enkf.enums import EnKFFSType
from res.test import ErtTestContext


@pytest.mark.equinor_test
class EnKFFSTest(ResTest):
    def setUp(self):
        self.mount_point = "storage/default"
        self.config_file = self.createTestPath("Equinor/config/with_data/config")

    def test_id_enum(self):
        self.assertEnumIsFullyDefined(
            EnKFFSType, "fs_driver_impl", "lib/include/ert/enkf/fs_types.hpp"
        )

    def test_create(self):
        with TestAreaContext("create_fs") as work_area:
            work_area.copy_parent_content(self.config_file)

            self.assertTrue(EnkfFs.exists(self.mount_point))
            fs = EnkfFs(self.mount_point)
            self.assertEqual(1, fs.refCount())
            fs.umount()

            self.assertFalse(EnkfFs.exists("newFS"))
            fs = EnkfFs.createFileSystem("newFS")
            self.assertTrue(EnkfFs.exists("newFS"))
            self.assertTrue(fs is None)

            with self.assertRaises(IOError):
                version = EnkfFs.diskVersion("does/not/exist")

            version = EnkfFs.diskVersion("newFS")
            self.assertTrue(version >= 106)

    @tmpdir()
    def test_create2(self):
        with TestAreaContext("create_fs2") as work_area:
            work_area.copy_parent_content(self.config_file)

            new_fs = EnkfFs.createFileSystem("newFS", mount=True)
            self.assertTrue(isinstance(new_fs, EnkfFs))

    def test_throws(self):
        with self.assertRaises(Exception):
            fs = EnkfFs("/does/not/exist")
