from eth_account import Account
from web3 import Web3

# Function to get the balance of an address
def get_balance(address, infura_url):
    web3 = Web3(Web3.HTTPProvider(infura_url))
    if not web3.is_connected():
        raise Exception("Failed to connect to Ethereum node")
    balance_wei = web3.eth.get_balance(address)
    balance_eth = web3.fromWei(balance_wei, 'ether')
    return balance_eth

# Infura URL (replace YOUR_INFURA_PROJECT_ID with your actual Project ID)
infura_url = "https://mainnet.infura.io/v3/1234567890abcdef1234567890abcdef"


# Generate a new private key and address
new_account = Account.create()
private_key = new_account.key.hex()
address = new_account.address

# Get the balance of the generated address
balance = get_balance(address, infura_url)

print(f"Private Key: {private_key}")
print(f"Address: {address}")
print(f"Balance: {balance} ETH")
