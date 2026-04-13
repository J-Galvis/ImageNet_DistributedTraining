# Machine configuration for Vms

### Install python, pip, venv & git
```
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```
### Configure environment
```
mkdir testing
cd testing
git init
git remote add origin https://github.com/J-Galvis/DistributedTrainnig_3NN.git
git pull origin master
```
### Create venv
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start server
```
python3 Distributed_NN/Server.py --inserrNeededFlags
```

```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣤⣀⠀⠀⠀⠀⠀⠀⠀⣠⣤⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⡟⠁⠀⢻⡄⠀⠀⠀⠀⠀⣸⠋⠀⠹⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣄⣠⣾⡧⠀⠀⠀⠀⠀⣿⣦⣀⣼⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⡧⠄⠀⠀⠀⠀⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⠏⠀⠀⠀⠀⠀⣿⡻⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣠⠖⠋⠉⠉⠓⢦⡀⠀⢿⣤⣴⡿⠀⠀⠀⠀⠀⠀⠸⣷⣤⣼⠇⠀⢀⣠⠤⠀⠠⠤⣄⡀
⠷⣄⠘⠀⠀⣢⣠⠟⠀⠀⠉⠋⠀⠰⣦⣀⣀⣀⣶⠀⠉⠉⠉⠀⠀⠸⣄⠀⠠⠶⠤⣀⡷
⠀⠀⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠒⠒⠚⠋⠀
```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀