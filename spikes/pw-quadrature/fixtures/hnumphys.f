c Information Technology Laboratory, National Institute of 
c Standards and Technology, 325 Broadway, Mail Stop 891.01, 
c Boulder, CO 80305-3328. email: zydrunas.gimbutas@nist.gov. 
c Contributions by staff of NIST, an agency of the U.S. Government, 
c are not subject to copyright within the United States.

c NIST-developed software is provided by NIST as a public service. 
c You may use, copy, and distribute copies of the software in any medium, 
c provided that you keep intact this entire notice. You may improve, 
c modify, and create derivative works of the software or any portion of 
c the software, and you may copy and distribute such 
c modifications or works. Modified works should carry a notice stating 
c that you changed the software and should note the date and nature 
c of any such change. Please explicitly acknowledge the National 
c Institute of Standards and Technology as the source of the software. 

c NIST-developed software is expressly provided "AS IS." 
c NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT, OR 
c ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE 
c IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, 
c NON-INFRINGEMENT, AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR 
c WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED 
c OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT
c WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE 
c SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE 
c CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
c
c You are solely responsible for determining the appropriateness of 
c using and distributing the software and you assume all risks 
c associated with its use, including but not limited to the risks 
c and costs of program errors, compliance with applicable laws, damage 
c to or loss of data, programs or equipment, and the unavailability 
c or interruption of operation. This software is not intended to be 
c used in any situation where a failure could cause risk of injury 
c or damage to property. The software developed by NIST employees is 
c not subject to copyright protection within the United States.


c Changed on 2026-11-02 for the fmm repository (Helmholtz CH0.4): cut by
c spikes/pw-quadrature/tools/extract_fixture.sh from FMM3D v2.1.0,
c src/Helmholtz/hnumphys.f, to domains 1 11 139 172 236 only (in hnumfour.f
c and hnumphys.f also to the routines of iprec 2 and 3); otherwise unchanged.
c
      subroutine hnumphys_iprec2(idomain,nl,nphys)
        if(idomain.eq.  1) then
          nphys(  1) =   5
          nphys(  2) =   9
          nphys(  3) =  13
          nphys(  4) =  16
          nphys(  5) =  20
          nphys(  6) =  24
          nphys(  7) =  28
          nphys(  8) =  33
          nphys(  9) =  38
          nphys( 10) =  42
          nphys( 11) =  46
          nphys( 12) =  50
          nphys( 13) =  54
          nphys( 14) =  61
          nphys( 15) =  64
          nphys( 16) =  66
          nphys( 17) =  70
          nphys( 18) =  73
          nphys( 19) =  41
          nphys( 20) =  18
          nphys( 21) =   8
        endif
        if(idomain.eq. 11) then
          nphys(  1) =  11
          nphys(  2) =  18
          nphys(  3) =  23
          nphys(  4) =  30
          nphys(  5) =  35
          nphys(  6) =  41
          nphys(  7) =  44
          nphys(  8) =  49
          nphys(  9) =  53
          nphys( 10) =  57
          nphys( 11) =  61
          nphys( 12) =  65
          nphys( 13) =  68
          nphys( 14) =  72
          nphys( 15) =  74
          nphys( 16) =  78
          nphys( 17) =  81
          nphys( 18) =  82
          nphys( 19) =  85
          nphys( 20) =  87
          nphys( 21) =  89
          nphys( 22) =  90
          nphys( 23) =  92
          nphys( 24) =  93
          nphys( 25) =  94
          nphys( 26) =  94
          nphys( 27) =  97
          nphys( 28) =  98
          nphys( 29) = 101
          nphys( 30) = 102
          nphys( 31) = 103
          nphys( 32) = 104
          nphys( 33) =  77
          nphys( 34) =  68
          nphys( 35) =  11
        endif
        if(idomain.eq.139) then
          nphys(  1) =   9
          nphys(  2) =  14
          nphys(  3) =  18
          nphys(  4) =  22
          nphys(  5) =  26
          nphys(  6) =  30
          nphys(  7) =  37
          nphys(  8) =  41
          nphys(  9) =  45
          nphys( 10) =  50
          nphys( 11) =  53
          nphys( 12) =  57
          nphys( 13) =  61
          nphys( 14) =  65
          nphys( 15) =  69
          nphys( 16) =  73
          nphys( 17) =  74
          nphys( 18) =  15
          nphys( 19) =  26
          nphys( 20) =   7
        endif
        if(idomain.eq.172) then
          nphys(  1) =  10
          nphys(  2) =  16
          nphys(  3) =  22
          nphys(  4) =  27
          nphys(  5) =  31
          nphys(  6) =  36
          nphys(  7) =  40
          nphys(  8) =  44
          nphys(  9) =  48
          nphys( 10) =  52
          nphys( 11) =  55
          nphys( 12) =  58
          nphys( 13) =  62
          nphys( 14) =  65
          nphys( 15) =  67
          nphys( 16) =  71
          nphys( 17) =  72
          nphys( 18) =  74
          nphys( 19) =  77
          nphys( 20) =  79
          nphys( 21) =  81
          nphys( 22) =  84
          nphys( 23) =  86
          nphys( 24) =  88
          nphys( 25) =  91
          nphys( 26) =  92
          nphys( 27) =  94
          nphys( 28) =  95
          nphys( 29) =  97
          nphys( 30) =  99
          nphys( 31) = 101
          nphys( 32) = 102
          nphys( 33) =  46
          nphys( 34) =   7
          nphys( 35) =   3
          nphys( 36) =   1
          nphys( 37) =   1
        endif
        if(idomain.eq.236) then
          nphys(  1) =   6
          nphys(  2) =  13
          nphys(  3) =  18
          nphys(  4) =  24
          nphys(  5) =  30
          nphys(  6) =  35
          nphys(  7) =  41
          nphys(  8) =  44
          nphys(  9) =  47
          nphys( 10) =  51
          nphys( 11) =  14
          nphys( 12) =  13
          nphys( 13) =   5
          nphys( 14) =  13
        endif
      subroutine hnumphys_iprec3(idomain,nl,nphys)
        if(idomain.eq.  1) then
          nphys(  1) =   6
          nphys(  2) =   9
          nphys(  3) =  12
          nphys(  4) =  15
          nphys(  5) =  18
          nphys(  6) =  22
          nphys(  7) =  26
          nphys(  8) =  30
          nphys(  9) =  34
          nphys( 10) =  38
          nphys( 11) =  42
          nphys( 12) =  46
          nphys( 13) =  50
          nphys( 14) =  54
          nphys( 15) =  58
          nphys( 16) =  62
          nphys( 17) =  67
          nphys( 18) =  71
          nphys( 19) =  75
          nphys( 20) =  79
          nphys( 21) =  83
          nphys( 22) =  87
          nphys( 23) =  91
          nphys( 24) =  95
          nphys( 25) =  99
          nphys( 26) = 103
          nphys( 27) = 107
          nphys( 28) = 111
          nphys( 29) = 112
          nphys( 30) =  29
          nphys( 31) =  28
          nphys( 32) =  14
          nphys( 33) =   2
        endif
        if(idomain.eq. 11) then
          nphys(  1) =  14
          nphys(  2) =  21
          nphys(  3) =  27
          nphys(  4) =  33
          nphys(  5) =  38
          nphys(  6) =  42
          nphys(  7) =  47
          nphys(  8) =  53
          nphys(  9) =  57
          nphys( 10) =  61
          nphys( 11) =  65
          nphys( 12) =  68
          nphys( 13) =  72
          nphys( 14) =  76
          nphys( 15) =  79
          nphys( 16) =  82
          nphys( 17) =  86
          nphys( 18) =  86
          nphys( 19) =  90
          nphys( 20) =  92
          nphys( 21) =  94
          nphys( 22) =  96
          nphys( 23) =  97
          nphys( 24) =  98
          nphys( 25) =  99
          nphys( 26) = 102
          nphys( 27) = 103
          nphys( 28) = 103
          nphys( 29) = 105
          nphys( 30) = 106
          nphys( 31) = 110
          nphys( 32) = 112
          nphys( 33) = 114
          nphys( 34) = 117
          nphys( 35) = 121
          nphys( 36) = 123
          nphys( 37) = 125
          nphys( 38) = 129
          nphys( 39) = 131
          nphys( 40) = 134
          nphys( 41) = 132
          nphys( 42) = 106
          nphys( 43) =  51
          nphys( 44) =   2
        endif
        if(idomain.eq.139) then
          nphys(  1) =  10
          nphys(  2) =  15
          nphys(  3) =  19
          nphys(  4) =  22
          nphys(  5) =  26
          nphys(  6) =  30
          nphys(  7) =  34
          nphys(  8) =  38
          nphys(  9) =  42
          nphys( 10) =  46
          nphys( 11) =  50
          nphys( 12) =  54
          nphys( 13) =  58
          nphys( 14) =  62
          nphys( 15) =  66
          nphys( 16) =  70
          nphys( 17) =  74
          nphys( 18) =  78
          nphys( 19) =  82
          nphys( 20) =  86
          nphys( 21) =  90
          nphys( 22) =  94
          nphys( 23) =  98
          nphys( 24) = 102
          nphys( 25) = 106
          nphys( 26) = 109
          nphys( 27) = 113
          nphys( 28) = 110
          nphys( 29) =  11
          nphys( 30) =  46
          nphys( 31) =   6
          nphys( 32) =   1
        endif
        if(idomain.eq.172) then
          nphys(  1) =  13
          nphys(  2) =  19
          nphys(  3) =  25
          nphys(  4) =  29
          nphys(  5) =  34
          nphys(  6) =  41
          nphys(  7) =  43
          nphys(  8) =  47
          nphys(  9) =  50
          nphys( 10) =  54
          nphys( 11) =  58
          nphys( 12) =  61
          nphys( 13) =  65
          nphys( 14) =  66
          nphys( 15) =  70
          nphys( 16) =  73
          nphys( 17) =  74
          nphys( 18) =  77
          nphys( 19) =  79
          nphys( 20) =  81
          nphys( 21) =  82
          nphys( 22) =  85
          nphys( 23) =  86
          nphys( 24) =  89
          nphys( 25) =  90
          nphys( 26) =  91
          nphys( 27) =  93
          nphys( 28) =  95
          nphys( 29) =  97
          nphys( 30) =  98
          nphys( 31) = 100
          nphys( 32) = 102
          nphys( 33) = 104
          nphys( 34) = 106
          nphys( 35) = 108
          nphys( 36) = 110
          nphys( 37) = 113
          nphys( 38) = 117
          nphys( 39) = 119
          nphys( 40) = 122
          nphys( 41) = 123
          nphys( 42) = 127
          nphys( 43) = 129
          nphys( 44) = 133
          nphys( 45) = 133
          nphys( 46) =  17
          nphys( 47) =  13
          nphys( 48) =  10
          nphys( 49) =   1
          nphys( 50) =   1
          nphys( 51) =   1
          nphys( 52) =   1
        endif
        if(idomain.eq.236) then
          nphys(  1) =   9
          nphys(  2) =  15
          nphys(  3) =  21
          nphys(  4) =  26
          nphys(  5) =  33
          nphys(  6) =  38
          nphys(  7) =  43
          nphys(  8) =  48
          nphys(  9) =  52
          nphys( 10) =  59
          nphys( 11) =  62
          nphys( 12) =  69
          nphys( 13) =  72
          nphys( 14) =  77
          nphys( 15) =  81
          nphys( 16) =  85
          nphys( 17) =  89
          nphys( 18) =  93
          nphys( 19) =  96
          nphys( 20) =  97
          nphys( 21) =  41
          nphys( 22) =  16
          nphys( 23) =  11
          nphys( 24) =   3
          nphys( 25) =   1
          nphys( 26) =   2
        endif
