import tempfile
from pydub import AudioSegment
from urllib.request import urlopen

data = urlopen('https://freesound.org/people/Argande102/sounds/170439/download/170439__argande102__wind-on-microphone.mp3').read()
f = tempfile.NamedTemporaryFile(delete=False)
f.write(data)
AudioSegment.from_mp3(f.name).export('result.ogg', format='ogg')
f.close()