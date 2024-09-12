# Before running:

- Expose all ports at CPs (use zone/source-ip):
    - sudo firewall-cmd --zone=trusted --add-source=142.104.69.241
    - sudo firewall-cmd --zone=trusted --add-source=142.104.69.242
    - sudo firewall-cmd --zone=trusted --add-source=142.104.69.237
    - sudo firewall-cmd --zone=trusted --add-source=142.104.69.238

- Also, ssh-keygen and ssh-copy-id from within container to each party
- Also, COPY input/* does not work for some reason. Make sure to populate Player-Data/Input-P0-0 with 8192 entries

- Run Scripts/compile-run.py -R 256 -H hosts.txt dealer-ring <program> in container
