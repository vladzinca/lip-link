Back log for Lip Link
----------

To-do legend: [*] improved; [+] feature; [-] fixed; [!] notes;

   - [+] Create lip-link-kernel class for messages.
   - [*] Check that the training uses the GPU.
   - [*] Instead of checking the size of the downloaded zip to check for its integrity, check the MD5 checksum.
   - [+] Use all the speakers from the GRID dataset.
   - [+] Use DLib face detector or a pre-trained YOLO to extract the mouth.
   - [*] The way the mean, standard deviation are calculated could be wrong. The mean should be calculated for every frame, probably.
   - [*] Check if the "'?!123456789" letters have to appear in vocabulary.
   - [+] Create unit tests.
   - [*] Add a pip list or pip freeze in the README file.
   - [*] Remove pre-commit from requirements after you are done.
