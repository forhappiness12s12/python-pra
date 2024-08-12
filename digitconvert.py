import base58

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
base58_address = 'TQQs3vL7MRKVKhJXCp7nvGPYETCsRpotS7'

# Convert Base58 to integer
integer_value = base58_to_int(base58_address)

# Convert integer to 10-digit decimal
ten_digit_decimal = int_to_10_digit_decimal(integer_value)

print(f'10-digit Decimal: {ten_digit_decimal}')
