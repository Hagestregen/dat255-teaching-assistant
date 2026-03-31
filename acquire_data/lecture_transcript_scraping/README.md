# Scraping transcript of video lectures

The auto generated transcript of the lectures where often in norwegian even though most of the lecture was in english. It seems to be really unaccurate.

Downloading and redoing the transcribation:
I didnt find a simple download button on panopto. I used this method to be able to download it as mp4:

1. Watch one of the lectures on panopto.
2. Right click and inspect
3. Go to the network tab and write "mp4" in the filter at the top.
4. refresh the page and double click the link if it shows up
5. Now right click and save as should work, and you are able to get it as mp4.

Note: On some web pages, m4a (MPEG-4 Audio) can be search for instead of mp4, which only contains sound, but didnt work here.

## Converting mp4 videofiles to mp3 sound files

Run the mp4_to_mp3.py script. Save the mp3 files in data/lecture_mp3/

The convert_all_mp4.bat or convert_all_mp4.bash will go through the data/lecture_mp4 folder and process all the mp4 files and out them in the data/lecture_mp3/ folder.
