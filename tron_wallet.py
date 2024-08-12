import tronpy
from tronpy import Tron
from tronpy.keys import PrivateKey
from tronpy.exceptions import AddressNotFound

# Generate a new private key
private_key = PrivateKey.random()
private_key_hex = private_key.hex()

# Derive the public address from the private key
public_address = private_key.public_key.to_base58check_address()

# Initialize Tron client
client = Tron()

# Initialize balance variable
balance_trx = 0

try:
    # Get account information
    account = client.get_account(public_address)
    # Extract balance (in Sun, where 1 TRX = 1,000,000 Sun)
    balance_sun = account['balance']
    balance_trx = balance_sun / 1_000_000  # convert balance to TRX
except AddressNotFound:
    print("Address not found on-chain. This is expected for a newly generated address.")

print(f"Private Key: {private_key_hex}")
print(f"Public Address: {public_address}")
print(f"Balance: {balance_trx} TRX")
