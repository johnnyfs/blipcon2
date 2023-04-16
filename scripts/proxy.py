import argparse
import requests

def post_data(url, payload):
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read data from a file and send a POST request.")
    parser.add_argument("-i", "--input", help="Input file", required=True)
    parser.add_argument("-o", "--output", help="Output file", required=True)
    parser.add_argument("-u", "--url", help="URL to post data to", required=True)
    parser.add_argument("-m", "--request-mapping", help="Mapping of string:size", required=True)
    parser.add_argument("-n", "--response-mapping", help="Mapping of string:size", required=True)

    args = parser.parse_args()

    # Convert the mapping string to a list of (string, size) tuples
    in_mapping = [pair.split(":") for pair in args.request_mapping.split(",")]
    out_mapping = [pair.split(":") for pair in args.response_mapping.split(",")]

    with open(args.input, "rb") as infile:
        with open(args.output, "wb") as outfile:
            while True:
                data = {}
                for key, size in in_mapping:
                    data[key] = infile.read(int(size))

                resp = requests.post(args.url, json=data)
                resp.raise_for_status()
                resp = resp.json()
                
                for key, size in out_mapping:
                    value = resp[key]
                    # If value iterable, write one value per size
                    if not hasattr(value, "__iter__"):
                        value = [value]
                    for v in value:
                            outfile.write(str(v).ljust(int(size)))
