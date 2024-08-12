import hashlib
import base58
from ecdsa import SigningKey, SECP256k1

def private_key_to_public_key(private_key_hex):
    private_key_bytes = bytes.fromhex(private_key_hex)
    sk = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
    vk = sk.get_verifying_key()
    public_key_bytes = vk.to_string('compressed')
    return public_key_bytes

def public_key_to_address(public_key_bytes):
    # Hash the public key with SHA-256 and then RIPEMD-160
    sha256_hash = hashlib.sha256(public_key_bytes).digest()
    ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
    
    # Add the Tron prefix (0x41)
    tron_prefix = b'\x41'
    prefixed_hash = tron_prefix + ripemd160_hash
    
    # Compute the checksum
    checksum = hashlib.sha256(hashlib.sha256(prefixed_hash).digest()).digest()[:4]
    
    # Create the final address by appending the checksum
    address_bytes = prefixed_hash + checksum
    
    # Encode the address using Base58Check encoding
    address = base58.b58encode(address_bytes)
    return address.decode()

# Example private key (replace with your actual private key)
private_key_hex = 'b528664c362bdda9e40d0f70772514ab7808532b24e74deeca0178cd562e3bc6'

# Get and print the Tron public address
public_key_bytes = private_key_to_public_key(private_key_hex)
tron_address = public_key_to_address(public_key_bytes)
print(f'Tron Address: {tron_address}')
def base58_to_int(base58_str):
    # Decode Base58 to bytes
    decoded_bytes = base58.b58decode(base58_str)
    
    # Convert bytes to integer
    integer_value = int.from_bytes(decoded_bytes, byteorder='big')
    
    return integer_value

def int_to_10_digit_decimal(integer_value):
    # Convert integer to a 10-digit decimal number
    return f'{integer_value:010}'

# Base58-encoded Tron address
base58_address = tron_address

# Convert Base58 to integer
integer_value = base58_to_int(base58_address)

# Convert integer to 10-digit decimal
ten_digit_decimal = int_to_10_digit_decimal(integer_value)

print(f'10-digit Decimal: {ten_digit_decimal}')
