#!/usr/bin/env python

import os
import sys
import glob
import zipfile

SRC_FILES       = ('jni/*.cpp', 'jni/*.hpp', 'jni/*.h', 'jni/ar/*')
REPORT_PATH     = 'report.pdf'
SUBMISSION_TAG  = 'project-2'

class SubmissionError(Exception): pass

def get_submission_files():
    src_files = reduce(list.__add__, (glob.glob(wc) for wc in SRC_FILES))
    if not src_files:
        raise SubmissionError('Source files not found! Execute this script in the project directory.')
    if not os.path.exists(REPORT_PATH):
        raise SubmissionError('Report not found! Make sure you have %s in the project directory.'%REPORT_PATH)
    return src_files+[REPORT_PATH]

def archive_submission(files):
    name = raw_input('Enter your SUNet ID (____@stanford.edu): ')
    name = name.split('@')[0].strip()
    if len(name)==0:
        raise SubmissionError('SUNet ID not provided.')
    filename = '-'.join([name, SUBMISSION_TAG])+'.zip'
    with zipfile.ZipFile(filename, 'w') as archive:
        for path in files:
            archive.write(path)
    return filename

def main():
    try:
        submission_files = get_submission_files()
        filename = archive_submission(submission_files)
        print('%s successfully created.'%filename)
        print('Remember to email it to the course staff as instructed!')
    except SubmissionError as e:
        print(e)
        print('Failed to create submission. Try again.')
        sys.exit(-1)

if __name__=='__main__':
    main()
