# Shazart
Ever stared at a masterpiece in a museum and thought, "It’s pretty, but I can't bother to read the tedious description?" Shazart is the Shazam for the art world—with a personality. Search by title or by an image, and we’ll skip the dry, crusty textbook jargon. Instead, you get a punchy, fun retelling of the scandals, the secrets, and the "what was the artist thinking?" behind the canvas.
Ask question about the collection. The RAG augmented system will be able to answer your questions!

# How to run 
Download the entire collection of the Rijkmuseum with the public API running 
```
uv run data.py
```
Query the model about the collection with the agentic ai system via streamlit
```
streamlit run app.py
```

# Build as a docker container
```
docker build -t streamlit_demo
```
# Live demo
A live demo is available on my server. [check it out](https:://streamlit.frabeamar.uk)

