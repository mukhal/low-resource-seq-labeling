#!/bin/bash
tail -c "$(($(stat -c%s - < "$2") * $1 / 100))" < "$2"
