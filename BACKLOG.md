Back log for Lip Link
----------

To-do legend: [*] improved; [+] feature; [-] fixed; [!] notes;

   - [+] Create lip-link-kernel class for messages.
   - [*] Remove one frame from 75 instead of padding 74 frames to 75. Make sure you are not adding the blank frame in the beginning instead of the end.
   - [*] Check that the training uses the GPU.
   - [*] Instead of checking the size of the downloaded zip to check for its integrity, check the MD5 checksum.
   - [*] Try to use os directly to find the name of a file from a path, instead of splitting strings like that.
   - [*] Consider using os.walk instead of glob.glob for bringing large amounts of data into RAM.
   - [+] Use all the speakers from the GRID dataset.
   - [+] Use DLib face detector or a pre-trained YOLO to extract the mouth.
   - [*] The way the mean, standard deviation are calculated could be wrong. The mean should be calculated for every frame, probably.
   - [*] Check if the "'?!123456789" letters have to appear in vocabulary.
   - [*] Add _ to mark methods that are not meant to be used externally.
   - [+] Create unit tests.
   - [*] Add a pip list or pip freeze in the README file.
   - [*] Remove pre-commit from requirements after you are done.
