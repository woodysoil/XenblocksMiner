from passlib.hash import argon2

# Function to generate an Argon2id hash
def generate_argon2id_hash(password):
    # Define Argon2id parameters
    t_cost = 1  # Time cost
    m_cost = 1 << 16  # Memory cost
    parallelism = 1  # Parallelism degree

    # Generate a salt (for this example, we'll just use a fixed salt)
    salt = bytes.fromhex("24691E54aFafe2416a8252097C9Ca67557271475")

    # Create a PasswordHasher instance with specified parameters
    ph = argon2.using(time_cost=t_cost, salt=salt, memory_cost=m_cost, parallelism=parallelism, hash_len = 64)

    # Generate the hash
    return ph.hash(password)

# Function to verify a password using Argon2id
def verify_argon2id_hash(password, hashed_password):
    try:
        # Verify the password
        return argon2.verify(hashed_password, password)
    except Exception:
        return False

# Example usage
password = "woody"  # Replace with your password to be hashed and verified

# Generate an Argon2id hash
hashed_password = generate_argon2id_hash(password)
print("Generated Hash:", hashed_password)

# Verify the password
if verify_argon2id_hash(password, hashed_password):
    print("Password verified!")
else:
    print("Invalid password!")
