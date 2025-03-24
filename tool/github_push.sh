#!/usr/bin/bash

echo -n "##### AIT GITHUB PUSH #####"
echo -n "*************************************************************************************"
echo -n "*************************************************************************************"
echo -n "**  important :                                                                    **"
echo -n "**    Before publishing the AIT to GITHUB,                                         **"
echo -n "**    ensure that no personal or confidential information is disclosed and         **"
echo -n "**    that the publication aligns with your objectives.                            **"
echo -n "**    Please proceed with these precautions.                                       **"
echo -n "**                                                                                 **"
echo -n "**  Disclaimer :                                                                   **"
echo -n "**    The publication of AITs on platform is the                                   **"
echo -n "**    responsibility of the developers.                                            **"
echo -n "**    The developers guarantee that the information contained in AIT               **"
echo -n "**    does not infringe any third-party intellectual property rights and           **"
echo -n "**    does not include any personal or confidential information.                   **"
echo -n "**                                                                                 **"
echo -n "**    In the event that developers inadvertently publish confidential information, **"
echo -n "**    or if AIT is improperly used by a third party,                               **"
echo -n "**    neither this platform nor its operators shall bear any liability.            **"
echo -n "**    The developers shall bear full responsibility for any direct or              **"
echo -n "**    indirect damages that may arise from the use of AIT.                         **"
echo -n "**                                                                                 **"
echo -n "**    Users of this platform understand the potential risks associated with        **"
echo -n "**    using AIT and agree to use AIT at their own risk.                            **"
echo -n "*************************************************************************************"
echo -n "*************************************************************************************"

INPUT_INVENTORY_ADD_FLAG=""
INPUT_COMMIT_COMMENT=""
COMMIT_ID=""
REPO_URL=""
REPO_NAME=""
GITHUB_ACCOUNT=""
CLONE_TYPE=""

echo -n "--- Step-1 Inventory upload? (Y:upload N:not upload) : "

while true
do
  read yn
  if [ "$yn" = "y" -o "$yn" = "Y" ]; then
    echo "Selected Inventory upload : Y"
    INPUT_INVENTORY_ADD_FLAG="Y"
    break
  elif [ "$yn" = "n" -o "$yn" = "N" ]; then
    echo "Selected Inventory upload : N"
    INPUT_INVENTORY_ADD_FLAG="N"
    break
  else 
    echo "Invalid input. Enter Y or N."
  fi
done

read -p "--- Step-2 Input commit comment." INPUT_COMMIT_COMMENT

echo Inputed commit comment : $INPUT_COMMIT_COMMENT

cd ..

REPO_URL=$(git remote get-url origin)

if [ "`echo $REPO_URL | grep 'https' `" ]; then
 CLONE_TYPE="H"
 GITHUB_ACCOUNT="`echo $REPO_URL | cut -d '/' -f 4`"
 REPO_NAME="`echo $REPO_URL | cut -d '/' -f 5`"
else 
 CLONE_TYPE="S"
 GITHUB_ACCOUNT_TEMP="`echo $REPO_URL | cut -d '/' -f 1`"
 GITHUB_ACCOUNT="`echo $GITHUB_ACCOUNT_TEMP | cut -d ':' -f 2`"
 REPO_NAME="`echo $REPO_URL | cut -d '/' -f 2`"
fi

if [ "$CLONE_TYPE" = "S" ]; then
 git remote set-url origin git@github.com:$GITHUB_ACCOUNT/$REPO_NAME
fi

git add deploy
git add develop
git add tool
git add LICENSE.txt 
git add readme.md 
git add ThirdPartyNotices.txt

if [ "$INPUT_INVENTORY_ADD_FLAG" = "Y" ]; then 
 git add local_qai/inventory
fi

git commit -m $INPUT_COMMIT_COMMENT

git push origin main

COMMIT_ID=$(git show --format="%H" --no-patch)

echo "------------------ Repository URL Start ------------------"
echo "https://github.com/$GITHUB_ACCOUNT/$REPO_NAME/tree/$COMMIT_ID"
echo "------------------ Repository URL End ------------------"

