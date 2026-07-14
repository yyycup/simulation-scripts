import unittest

from btms_runtime import ensure_env_library_bin_on_path


class BtmsRuntimeTest(unittest.TestCase):
    def test_prepends_active_conda_library_bin_once(self):
        environ = {"PATH": r"C:\Windows\System32"}
        library_bin = ensure_env_library_bin_on_path(
            python_executable=r"C:\Users\demo\miniforge3\envs\btms\python.exe",
            environ=environ,
        )
        ensure_env_library_bin_on_path(
            python_executable=r"C:\Users\demo\miniforge3\envs\btms\python.exe",
            environ=environ,
        )

        self.assertEqual(str(library_bin), r"C:\Users\demo\miniforge3\envs\btms\Library\bin")
        self.assertEqual(environ["PATH"].split(";")[0], str(library_bin))
        self.assertEqual(environ["PATH"].lower().count(str(library_bin).lower()), 1)


if __name__ == "__main__":
    unittest.main()
