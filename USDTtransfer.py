import tronpy
from tronpy.keys import PrivateKey

# Connect to Tron blockchain
client = tronpy.Tron()

# Replace these with your private key and the recipient's address
private_key_hex = '0000000000000000000000000000000000000000000000000000000000000002'
recipient_address = 'TE7g56Z23iTZ5ZYmqacKRvDMkF7VoULsWy'
usdt_contract_address = 'TXLAQ63Xg1NAzckPwKHvzw7CSEmLMEqcdj'  # USDT contract address on Tron

# Replace this with the amount of USDT to send (in smallest units, 1 USDT = 1,000,000 micro-USD)
amount_in_usdt = 100  # Sending 1 USDT

# Create a private key object
private_key = PrivateKey(bytes.fromhex(private_key_hex))

# Get the sender's address from the private key
sender_address = private_key.public_key.to_base58check_address()

# Get the contract object
contract = client.get_contract(usdt_contract_address)

# Create a transaction to transfer USDT
txn = (
    contract.functions.transfer(recipient_address, amount_in_usdt)
    .with_owner(sender_address)
    .fee_limit(1_000_000)  # Set appropriate fee limit
    .build()
    .sign(private_key)
)

# Broadcast the transaction
result = txn.broadcast()

# Print transaction result
print(f'Transaction Result: {result}')
