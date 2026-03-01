#Objective: Remove the "Zone.Identifier" alternate data stream from files in a specified directory to prevent security warnings on Windows.
# Check if directory is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
    exit 1
fi
DIRECTORY=$1
# Find and remove Zone.Identifier alternate data streams
find "$DIRECTORY" -type f -exec sh -c 'for file; do
    if [ -e "$file:Zone.Identifier" ]; then
        echo "Removing Zone.Identifier from $file"
        rm "$file:Zone.Identifier"
    fi
done' sh {} +
echo "Zone.Identifier streams removed from files in $DIRECTORY"