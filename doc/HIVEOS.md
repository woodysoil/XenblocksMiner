# How to prepare HIVEOS to mine xenblocks (nvidia GPU only!)

For those who are using AMD GPUs you can use old miner (given the low efficienty using opencl, you can set dual mining on these cards) link is here -> https://github.com/levykrak/xengpuminer



<br><hr>

Log into your account on hiveos farm. If you don't have a wallet created, make one <br>
![300184088-754e3334-0d91-4e84-82d2-b37b5acf985c](https://github.com/levykrak/XenblocksMiner/assets/44068840/a692e5f0-8750-462c-9530-6163e7286dbd)

<br><hr>

Create new flightsheet, pick coin xenblock, your wallet and choose miner as custom
<br><br>
![1flighsheet_create](https://github.com/levykrak/XenblocksMiner/assets/44068840/b99a33fb-714b-4a4a-9229-52928b26c830) 
<br>
<br>
Next, put link 
```bash
https://github.com/woodysoil/XenblocksMiner/releases/download/v1.3.1/xenblocksMiner-1.3.1_hiveos.tar.gz
```
into installation URL, change hash algorithm to "argon2id", fill up "Wallet and worker template" and "Pool URL", apply changes 
<br><br>

![custom_configuration](https://github.com/levykrak/XenblocksMiner/assets/44068840/c1d3ff81-b459-46b3-90d3-55d75e2a5790)

<br>
Click "create flight sheet" and active flight sheet for rig 
<br>

![flighsheet_click](https://github.com/levykrak/XenblocksMiner/assets/44068840/73eeee23-c07d-4ab8-a393-c4426b44a0eb)

<br><br><br>
HIVEOS will download package automatically, unpack it, and put it proper folder. It will take some time. After a while you should get information about mining

![finish](https://github.com/levykrak/XenblocksMiner/assets/44068840/f1ed1247-3788-4fb1-8489-b5d0269d9b64)

# For advance user

In the case of having multiple cards and mining multiple tokens on one rig, it is possible to select the gpu_id to be run by the miner. Simply add in json format devices you want to mine "device_id": "0,1,2,3" in extra config arguments
```bash
{"devfee_permillage": "40", "device_id": "0,2,4,6"}
```
Then you can add second miner and get something like this:
![obraz](https://github.com/levykrak/XenblocksMiner/assets/44068840/4bb25d7c-6349-4950-9783-ce3ae02e687c)



<br> <hr>
Standard miner is set to give 2,1% devfee for our work. If you think we don`t deserve for such work, you can adjust our fee (0-1000 one per mille). In that case you have to add <br>
```bash
{"devfee_permillage": "21"}
```
<br>
in "setup miner config" in your flightsheet

![obraz](https://github.com/levykrak/XenblocksMiner/assets/44068840/674ee713-26ec-4d43-a0a2-8430993600eb)





<br><br>
happy mining<br>
levykrak <br>
telegram @aklimczyk <br>
twitter @Arkadiu49209932

