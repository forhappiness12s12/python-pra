import tronpy
from tronpy.keys import PrivateKey
from tronpy import Contract

# USDT Contract ABI (example ABI)
USDT_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [
            {"name": "", "type": "bool"}
        ],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    }
    # You may need to include the full ABI for more complex interactions
]

# Connect to Tron blockchain
client = tronpy.Tron()

# Replace these with your private key and the recipient's address
private_key_hex = '0000000000000000000000000000000000000000000000000000000000000002'
recipient_address = 'TE7g56Z23iTZ5ZYmqacKRvDMkF7VoULsWy'
usdt_contract_address = 'TXLAQ63Xg1NAzckPwKHvzw7CSEmLMEqcdj'  # USDT contract address on Tron

# Replace this with the amount of USDT to send (in smallest units, 1 USDT = 1,000,000 micro-USD)
amount_in_usdt = 1000000  # Sending 1 USDT

# Create a private key object
private_key = PrivateKey(bytes.fromhex(private_key_hex))

# Get the sender's address from the private key
sender_address = private_key.public_key.to_base58check_address()

# Initialize the contract
contract = Contract(address=usdt_contract_address, client=client)

# Set the ABI for the contract
contract.abi = USDT_ABI

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
