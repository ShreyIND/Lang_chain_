from langchain_text_splitters import CharacterTextSplitter

text = "Your long document text goes here..."

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separator=""
)

chunks = splitter.split_text(text)
print(f"Total chunks: {len(chunks)}")
print(chunks[0])
