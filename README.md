### Setup instructions to run jupyter notebooks:

1. install python and pip / venv   
```sh
    # on Mac
    brew update    
    brew install python

    # on Ubuntu:  
    sudo apt update    
    sudo apt install python3-dev python3-pip python3-venv    
```
2. create a virtual environment  
```sh
    python3 -m venv --system-site-packages ./venv  
```
3. activate the venv  
```sh
    source ./venv/bin/activate  # sh, bash, or zsh  

    . ./venv/bin/activate.fish  # fish 

    source ./venv/bin/activate.csh  # csh or tcsh  
```
4. upgrade pip and list installed packages  
```sh
    pip install --upgrade pip  

    pip list  # show packages installed within the virtual environment 
```
5. install packages from requirements.txt  
```sh
    pip install -r requirements.txt  
```
6. run the jupyter notebooks in the browser  
```sh
    jupyter notebook #or  

    jupyter-notebook  
```
7. after deactivate the venv  
```sh
    deactivate
```  
