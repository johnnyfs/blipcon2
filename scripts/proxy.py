import argparse
import requests
import base64

description = """
Read raw data from a file, map it to specified request fields,
and send a POST request, then map specified response fields
to raw data and write it to another file.

The mapping strings are a comma-separated list of key:value pairs,
where key is the name of the field and value is the size of the field.
If the key's value is an iterable of length N, then N chunks of size
`size` will be written.
"""

def post_data(url, payload):
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", help="Input file", required=True)
    parser.add_argument("-o", "--output", help="Output file", required=True)
    parser.add_argument("-u", "--url", help="URL to post data to", required=True)
    parser.add_argument("-m", "--request-mapping", help="Mapping of string:size", required=True)
    parser.add_argument("-n", "--response-mapping", help="Mapping of string:size", required=True)

    args = parser.parse_args()

    # Convert the mapping string to a list of (string, size) tuples
    in_mapping = [pair.split(":") for pair in args.request_mapping.split(",")]
    out_mapping = [pair.split(":") for pair in args.response_mapping.split(",")]

    print("opening {} for reading".format(args.input))
    with open(args.input, "rb") as infile:
        print('Opened input file: ', args.input)
        with open(args.output, "wb") as outfile:
            print('Opened output file: ', args.output)
            while True:
                data = {}
                for key, size in in_mapping:
                    print(f"Reading {size} bytes for request key {key}...")
                    bytes_ = infile.read(int(size))
                    if int(size) <= 16:
                        print(f"Read {bytes_} for request key {key}...")
                    if len(bytes_) == 0:
                        print("Reached end of file, exiting...")
                        exit(0)
                    encoded = str(base64.encodebytes(bytes_), "utf-8")
                    data[key] = encoded

                print(f"Posting data to {args.url}...")
                resp = requests.post(args.url, json=data)
                resp.raise_for_status()
                resp = resp.json()
                
                for key, size in out_mapping:
                    value = resp[key]
                    print(f"Got response key {key} with value {value}...")
                    # If value iterable, write one value per size
                    if not hasattr(value, "__iter__"):
                        value = [value]
                    for v in value:
                        output = str(v).ljust(int(size)).encode("utf-8")
                        print(f"Writing {len(output)} bytes as or response key {key}...")
                        outfile.write(output)
                        outfile.flush()
