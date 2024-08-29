from yt_dlp import YoutubeDL
import ffmpeg
import os

def download_and_convert_from_yt(video_link,file_name='test'):

    # Define file names
    video_file_name = file_name+'_video_temp'
    audio_file_name= file_name+'_audio_temp'
    adjusted_audio_file_name = file_name+'_audio.wav'
    adjusted_video_file_name = file_name+'_video.mp4'
    # Define options for youtube-dl to download video
    video_opts = {
        'format': 'bestvideo+bestaudio/best',  # Download the best video and audio available
        'outtmpl': video_file_name,  # Output template for video file
        'postprocessors': [
            {'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',  # Convert video to mp4 after download
            }],
    }   

    # Define options for youtube-dl to extract audio separately
    audio_opts = {
        'format': 'bestaudio/best',  # Download the best audio available
        'outtmpl': audio_file_name,  # Temporary file for audio extraction
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }


    # Download the video
    with YoutubeDL(video_opts) as ydl:
        ydl.download([video_link])
    # Download the video
    with YoutubeDL(audio_opts) as ydl:
        ydl.download([video_link])

    # Convert the downloaded video to 30 FPS using ffmpeg

    ffmpeg.run(ffmpeg.output(ffmpeg.input(video_file_name+'.mp4').filter('fps',fps=30),adjusted_video_file_name))

    # Adjust the bitrate of the WAV file using ffmpeg
    ffmpeg.run(ffmpeg.output(ffmpeg.input(audio_file_name+'.wav'),adjusted_audio_file_name,af='aresample=16000'))

    # Clean up the original audio file if needed
    if os.path.isfile(video_file_name+'.mp4'):
        os.remove(video_file_name+'.mp4')
    else:
        print('You may need to manually clean up some temporary files')
    if os.path.isfile(audio_file_name+'.wav'):
        os.remove(audio_file_name+'.wav')
    else:
        print('You may need to manually clean up some temporary files')