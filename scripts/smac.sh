set -eu

TMPP=""
OLDPATH=$PATH
do_install=false

# Restore things back to normal.
function restore {
    export PATH=$OLDPATH
    if [[ "$TMPP" && -d "$TMPP" ]]; then
        rm -r $TMPP
    fi
}

if ! [ -x "$(command -v swig)" ] | [[ $(swig -version | grep "Version 3") ]]; then
    # Wrong version of swig, or not installed.
    # Try installing... if allowed.
    if $do_install; then
        sudo apt install -y swig
    fi

    if ! [[ $(swig -version | grep "Version 3") ]]; then
        echo "Default version of swig is not 3.0 ."
        # Check if swig 3.0 is available under a special name.
        # Try installing it otherwise.
        if ! [ -x "$(command -v swig3.0)" ]; then
            if $do_install; then
                echo "swig 3.0 was not installed. Installing."
                sudo apt install -y swig3.0\
            else
                echo "No swig 3.0 installation found... Aborting." 1>&2
                exit 1
            fi
        fi

        echo "Temporarily redirecting swig to swig 3.0"
        # Replace swig as found in path with swig 3.0 temporarily.
        # Create a symbolic link, and prefix it to the current path.
        TMPP=$(mktemp -d)

        if [[ ! "$TMPP" || ! -d "$TMPP" ]]; then
            echo "Could not create temp dir" >&2
            exit 1
        fi

        ln -s $(command -v swig3.0) $TMPP/swig
        export PATH=$TMPP:$PATH

        if ! [[ $(swig -version | grep "Version 3") ]]; then
            echo "Could not redirect swig." >&2
            exit 1
        fi

        trap restore EXIT
    fi
fi

# Install using pip?
if [ -x "$(command -v poetry)" ]; then
    poetry install -E smac
elif [ -x "$(command -v pip3)" ]; then
    pip3 install -r ${BASH_SOURCE[0]}/requiremaents_smac.txt
elif [ -x "$(command -v pip)" ]; then
    pip install -r ${BASH_SOURCE[0]}/requiremaents_smac.txt
else
    echo "No python package manager found..." 1>&2
    exit 1
fi

echo "Success! smac should now work."