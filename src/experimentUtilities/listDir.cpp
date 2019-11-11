#include <dirent.h>
#include <assert.h>
#include "listDir.h"

// My apologies for using this, as this will most likely not compile on Windows.
// Hopefully it's not too difficult to swap it out if you experience difficulties.

std::vector<std::string> listDir(std::string directory) {
	std::vector<std::string> directoryContents;

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(directory.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			std::string fileName(ent->d_name);
			if(fileName != "." && fileName != "..") {
				directoryContents.push_back(fileName);
			}
		}
		closedir (dir);
	} else {
		/* could not open directory */
		assert(false);
	}

    return directoryContents;
}