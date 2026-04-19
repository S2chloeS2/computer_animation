#
# Copyright (c) 2026. Columbia University. All rights reserved.
#
# This software and documentation contain confidential and proprietary
# information that is the property of Columbia University.
#
# Unauthorized copying, distribution, or modification of this file,
# via any medium, is strictly prohibited.
#
# Written by Changxi Zheng <cxz@cs.columbia.edu>, 2026
#
import nemo


def show():
    """Show a header in terminal"""
    print(r"""
                          +@@@@@=:@%-==
                          @*++++++*++++#
                          :#++++++++++++*
                          =##++++++++++++--  :.
                    =#%%##= :%*++++++++++@*.@#++++=:
                   =@*+++++*. %%+++++++++@ =@+++++++=.
                    =*++++++#: =@+++++++@#.@*++++....-:
     #@**+++=:        :+++++*%. .@@+++++@ :@++*:*@@@=  =
   +@@*+++++++++*%   #*++++++@=   *@%++*@ @+++*.@@@@@+ =+
  =@@*+++++++++++#@  %*++++++@@    %@%+@. @++++*.@@@@ .++=
  @@@*+++++++++++*@-:@*++++++@*.+++%@++@  %+++++++++++++++
  @@@***+++++++++*@=.@*++++++@-=+++++++@  @++++++++++++++=
  -@@#***++++++++%@  %*+++++#%.:@+++***+@ .@++++++=----:
   -@@%**+++++++*@=..=%+++++%-   #%++++++@ :%++++++=.
     -%%+=-:.        .=++++#=    @+++++++*@ :%*++++++-
                    =*+++++++=-:=*++++++++*@..%#++=:.
                    +@#+++++++-. ..:::::=**+++:
                     :#%*+=:.         =*+++**++.
                                      .+#*- :#-
""")
    print(f"Nemo {nemo.__version__}")
