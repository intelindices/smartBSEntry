"""Regenerate TesterFiles.mqh from the plugin catalog."""

from smartbs_engines.catalog import write_tester_files

if __name__ == "__main__":
    print("ok", write_tester_files())
