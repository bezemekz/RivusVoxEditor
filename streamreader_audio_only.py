from torchaudio.io import StreamReader

def stream(q, format, src, segment_length, sample_rate, NUM_ITER):
    print("Building StreamReader...")
    if format == None:
        streamer = StreamReader(src)
    else:
        streamer = StreamReader(src, format=format)
    streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=sample_rate, buffer_chunk_size =5) #it may be useful to play with buffer_chunk_size
    print(streamer.get_src_stream_info(0))
    print(streamer.get_out_stream_info(0))
    print("Streaming...")
    print()
    stream_iterator = streamer.stream(timeout=-1, backoff=1.0) #it may be useful to play with backoff
    for _ in range(NUM_ITER):
        (chunk,) = next(stream_iterator)
        q.put(chunk)