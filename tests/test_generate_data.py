import os, sys
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.settings as s
import pickle

# subprocess.check_output("ls " + os.path.join(s.ETL_DIR))

# print(os.path.join(s.ETL_DIR))

def test_generate_dataset():

	# X_train, X_test, y_train, y_test = generate()

	files = os.listdir(s.DATA_TRANSFORMED)
	assert "X_train.p" in files 
	assert "y_train.p" in files
	assert "X_test.p" in files
	assert "y_test.p" in files

