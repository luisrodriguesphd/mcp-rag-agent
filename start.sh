#!/bin/bash

REPO_NAME="mcp-rag-agent"
echo "Start setting up the $REPO_NAME repository"

echo "****************************************"
echo "Install pip using ensurepip and upgrade"
echo "****************************************"
python -m ensurepip --upgrade
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Failed to ensure the installation of pip for Python"
fi

echo "****************************************"
echo "Create and Activate Virtual Environment"
echo "****************************************"
python -m venv venv
source ./venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to create and activate virtual environment"
fi

echo "****************************************"
echo "Install Requirements"
echo "****************************************"
python -m pip install --upgrade pip
pip install -r requirements_dev.txt
pip install -e .
if [ $? -ne 0 ]; then
    echo "Failed to install requirements for development"
fi

echo "****************************************"
echo "Finish setting up the $REPO_NAME repository"
echo "****************************************"
