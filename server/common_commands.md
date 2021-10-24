# Find 5 largest directories or files

```du -a | sort -n -r | head -n 5
du -sh -- *  | sort -rh  Files and directories, or
du -sh -- */ | sort -rh  Directories only
```
