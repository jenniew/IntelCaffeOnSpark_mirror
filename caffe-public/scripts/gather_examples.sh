#!/bin/bash
# 
# All modification made by Intel Corporation: © 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Assemble documentation for the project into one directory via symbolic links.

# Find the docs dir, no matter where the script is called
ROOT_DIR="$( cd "$(dirname "$0")"/.. ; pwd -P )"
cd $ROOT_DIR

# Gather docs from examples/**/readme.md
GATHERED_DIR=docs/gathered
rm -r $GATHERED_DIR
mkdir $GATHERED_DIR
for README_FILENAME in $(find examples -iname "readme.md"); do
    # Only use file if it is to be included in docs.
    if grep -Fxq "include_in_docs: true" $README_FILENAME; then
        # Make link to readme.md in docs/gathered/.
        # Since everything is called readme.md, rename it by its dirname.
        README_DIRNAME=`dirname $README_FILENAME`
        DOCS_FILENAME=$GATHERED_DIR/$README_DIRNAME.md
        mkdir -p `dirname $DOCS_FILENAME`
        ln -s $ROOT_DIR/$README_FILENAME $DOCS_FILENAME
    fi
done

# Gather docs from examples/*.ipynb and add YAML front-matter.
for NOTEBOOK_FILENAME in $(find examples -depth -iname "*.ipynb"); do
    DOCS_FILENAME=$GATHERED_DIR/$NOTEBOOK_FILENAME
    mkdir -p `dirname $DOCS_FILENAME`
    python scripts/copy_notebook.py $NOTEBOOK_FILENAME $DOCS_FILENAME
done
