from torchaudio.io import StreamReader
import time

def stream(q, format, src, segment_length, sample_rate):

    print("Building StreamReader...")
    if format == None:
        streamer = StreamReader(src)
    else:
        streamer = StreamReader(src, format=format)
    streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=sample_rate, buffer_chunk_size =5) #it may be useful to play with buffer_chunk_size

    first_chunk_time = None
    num_chunks_put=0
    for (chunk,) in streamer.stream(timeout=-1):
        if first_chunk_time is None and chunk is not None:
            first_chunk_time = time.time()
        q.put([chunk,first_chunk_time])
        num_chunks_put+=1
        if (format==None)and(num_chunks_put>3):
            #simulate waiting for next audio chunk to come through if we are using a file. It seems with real audio the queue gets 3 chunks before the q starts getting things.
            time.sleep(segment_length/sample_rate)

    StreamReader.close()