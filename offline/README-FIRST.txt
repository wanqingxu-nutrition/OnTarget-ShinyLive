OnTarget - Diet Optimizer
Nutritional Quality and Cost Calculator for Hypothetical Diets in Boston

Built by Wanqing Xu under the supervision of William Masters,
Friedman School of Nutrition Science and Policy, Tufts University.
Contact: wanqing.xu@tufts.edu or william.masters@tufts.edu


HOW TO RUN IT
-------------
1. Unzip this entire folder somewhere on your computer
   (Desktop or Documents is fine). Keep all the files together.

2. Double-click  Start-OnTarget.bat

3. A black window opens and your browser follows a few seconds later.

4. The first load takes 30 to 90 seconds. The page is blank while it
   works - this is normal. It is starting a complete copy of Python
   inside your browser. Later runs are much faster.

5. Leave the black window open while you use the calculator.
   Closing it shuts the calculator down.


WHAT YOU NEED
-------------
Windows, and any modern browser (Chrome, Edge or Firefox).
Nothing to install. No Python. No internet connection.
The calculator and all its data run entirely on your own computer;
nothing you enter is sent anywhere.


IF IT DOES NOT START
--------------------
* "Windows protected your PC" blue box
  Click "More info", then "Run anyway". This appears because the file
  was downloaded from the internet, not because anything is wrong.

* Nothing happens when you double-click
  Right-click Start-OnTarget.bat, choose "Properties", and if there is
  an "Unblock" checkbox near the bottom, tick it and click OK.

* The browser says it cannot connect
  Make sure the black window is still open, then copy the address it
  shows (http://127.0.0.1:8765/) into your browser yourself.

* A blank page that never finishes loading
  Try a different browser. Chrome and Edge are the most reliable.
  Internet Explorer is not supported.


MAC OR LINUX
------------
Start-OnTarget.bat is Windows-only. On a Mac or Linux machine, open a
terminal in this folder and run:

    python3 -m http.server 8765 --directory site

then open http://127.0.0.1:8765/ in your browser.
