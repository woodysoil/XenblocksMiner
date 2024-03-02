#!/usr/bin/env bash

# levykrak has provided hiveos support for the old mining software xengpuminer (https://github.com/levykrak/xengpuminer),
# and has provided a hiveos environment that enabled us to offer hiveos support for the new miner xenblocksMiner, assisting in bug fixes.
# The default address below is used to express gratitude for these contributions to the ecosystem.

DEFAULT_ECODEV_ADDRESS="0xdc02E1efA76c24e9a4F35C77023cA010c875beB1" # levykrak's address
DEFAULT_DEVFEE_PERMILLAGE=21 # 2.1% for total devfee
# In the context of hiveos support, a 2% developer fee is quite standard and is intended to fuel further feature support and development. 
# This equates to a default devfee of 2.1% (21 per mille) for the total development fee.
# If you believe this developer fee is too high, we have provided an option for you to override it. 
# To do so, enter 'devfee_permillage=your_desired_value' in the 'extra config arguments' section of your flight sheet, 
# where 'your_desired_value' is a number between 0 and 1000, according to your preference.

DEVFEE_PERMILLAGE=$(echo $CUSTOM_USER_CONFIG | jq -r '.devfee_permillage // empty')
if [ -z "$DEVFEE_PERMILLAGE" ]; then
    DEVFEE_PERMILLAGE=$DEFAULT_DEVFEE_PERMILLAGE
fi

echo "devfee_permillage=$DEVFEE_PERMILLAGE
account_address=$CUSTOM_TEMPLATE" > $MINER_DIR/$CUSTOM_MINER/config.txt

ECODEV_ADDRESS=$(echo $CUSTOM_USER_CONFIG | jq -r '.ecodev_address // empty')
if [ -z "$ECODEV_ADDRESS" ]; then
    ECODEV_ADDRESS=$DEFAULT_ECODEV_ADDRESS
fi
if [ -n "$ECODEV_ADDRESS" ] && [ "$ECODEV_ADDRESS" != "null" ]; then
    echo "ecodev_address=$ECODEV_ADDRESS" >> $MINER_DIR/$CUSTOM_MINER/config.txt
fi