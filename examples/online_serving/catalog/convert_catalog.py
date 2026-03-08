import json
import sys


def process_json(input_file_path, output_file_path):
    try:
        with open(input_file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_file_path}'.")
        return

    result_list = []
    BASE = 8192  # 8K Base for sid2pid decode

    # Iterate through each key in the dictionary
    for key_str in data:
        try:
            # Convert key to integer
            key = int(key_str)
        except ValueError:
            print(f"Skipping non-integer key: {key_str}")
            continue

        # Calculate the values for s_a, s_b, s_c
        # 2. s_a: key / 8192 / 8192
        val_a = key // (BASE * BASE)

        # 3. s_b: (residual of key / 8192^2) / 8192
        # Residual of key / 8192^2 is (key % 8192^2)
        val_b = (key % (BASE * BASE)) // BASE

        # 4. s_c: residual of above (effectively key % 8192)
        val_c = key % BASE

        # Create the list of 5 tokens
        tokens = [
            "<|sid_begin|>",
            f"<s_a_{val_a}>",
            f"<s_b_{val_b}>",
            f"<s_c_{val_c}>",
            "<|sid_end|>",
        ]

        result_list.append(tokens)

    # Write the result to the output file
    with open(output_file_path, "w") as f:
        json.dump(result_list, f, indent=2)

    print(
        f"Successfully processed {len(result_list)} items. Output written to '{output_file_path}'."
    )


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        input_json = sys.argv[1]
        output_json = sys.argv[2]
    else:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    process_json(input_json, output_json)
