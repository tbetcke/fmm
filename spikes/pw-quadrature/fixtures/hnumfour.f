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
c src/Helmholtz/hnumfour.f, to domains 1 11 139 172 236 only (in hnumfour.f
c and hnumphys.f also to the routines of iprec 2 and 3); otherwise unchanged.
c
      subroutine hnumfour_iprec2(idomain,nl,nfour)
        if(idomain.eq.  1) then
          nfour(  1) =   1
          nfour(  2) =   3
          nfour(  3) =   4
          nfour(  4) =   5
          nfour(  5) =   6
          nfour(  6) =   7
          nfour(  7) =   7
          nfour(  8) =   8
          nfour(  9) =   8
          nfour( 10) =   9
          nfour( 11) =  10
          nfour( 12) =  10
          nfour( 13) =  10
          nfour( 14) =  10
          nfour( 15) =  10
          nfour( 16) =  10
          nfour( 17) =  10
          nfour( 18) =   9
          nfour( 19) =   7
          nfour( 20) =   9
          nfour( 21) =   5
        endif
        if(idomain.eq. 11) then
          nfour(  1) =   3
          nfour(  2) =   5
          nfour(  3) =   6
          nfour(  4) =   7
          nfour(  5) =   8
          nfour(  6) =   9
          nfour(  7) =  10
          nfour(  8) =  10
          nfour(  9) =  11
          nfour( 10) =  12
          nfour( 11) =  13
          nfour( 12) =  13
          nfour( 13) =  14
          nfour( 14) =  15
          nfour( 15) =  15
          nfour( 16) =  16
          nfour( 17) =  16
          nfour( 18) =  17
          nfour( 19) =  17
          nfour( 20) =  18
          nfour( 21) =  18
          nfour( 22) =  18
          nfour( 23) =  18
          nfour( 24) =  18
          nfour( 25) =  17
          nfour( 26) =  17
          nfour( 27) =  16
          nfour( 28) =  15
          nfour( 29) =  14
          nfour( 30) =  13
          nfour( 31) =  12
          nfour( 32) =   3
          nfour( 33) =   1
          nfour( 34) =   1
          nfour( 35) =   1
        endif
        if(idomain.eq.139) then
          nfour(  1) =   3
          nfour(  2) =   4
          nfour(  3) =   5
          nfour(  4) =   6
          nfour(  5) =   7
          nfour(  6) =   7
          nfour(  7) =   8
          nfour(  8) =   8
          nfour(  9) =   9
          nfour( 10) =  11
          nfour( 11) =  10
          nfour( 12) =  10
          nfour( 13) =  10
          nfour( 14) =  10
          nfour( 15) =  10
          nfour( 16) =  10
          nfour( 17) =   8
          nfour( 18) =   8
          nfour( 19) =   8
          nfour( 20) =   3
        endif
        if(idomain.eq.172) then
          nfour(  1) =   2
          nfour(  2) =   5
          nfour(  3) =   5
          nfour(  4) =   6
          nfour(  5) =   7
          nfour(  6) =   8
          nfour(  7) =   9
          nfour(  8) =   9
          nfour(  9) =  10
          nfour( 10) =  10
          nfour( 11) =  11
          nfour( 12) =  11
          nfour( 13) =  12
          nfour( 14) =  13
          nfour( 15) =  13
          nfour( 16) =  13
          nfour( 17) =  13
          nfour( 18) =  14
          nfour( 19) =  14
          nfour( 20) =  15
          nfour( 21) =  15
          nfour( 22) =  15
          nfour( 23) =  16
          nfour( 24) =  15
          nfour( 25) =  15
          nfour( 26) =  15
          nfour( 27) =  15
          nfour( 28) =  14
          nfour( 29) =  14
          nfour( 30) =  12
          nfour( 31) =  10
          nfour( 32) =  12
          nfour( 33) =  12
          nfour( 34) =   5
          nfour( 35) =   1
          nfour( 36) =   1
          nfour( 37) =   1
        endif
        if(idomain.eq.236) then
          nfour(  1) =   1
          nfour(  2) =   2
          nfour(  3) =   3
          nfour(  4) =   5
          nfour(  5) =   4
          nfour(  6) =   5
          nfour(  7) =   5
          nfour(  8) =   6
          nfour(  9) =   6
          nfour( 10) =   7
          nfour( 11) =   7
          nfour( 12) =  10
          nfour( 13) =   4
          nfour( 14) =  10
        endif
      subroutine hnumfour_iprec3(idomain,nl,nfour)
        if(idomain.eq.  1) then
          nfour(  1) =   2
          nfour(  2) =   3
          nfour(  3) =   4
          nfour(  4) =   5
          nfour(  5) =   6
          nfour(  6) =   7
          nfour(  7) =   8
          nfour(  8) =   9
          nfour(  9) =  10
          nfour( 10) =  11
          nfour( 11) =  11
          nfour( 12) =  12
          nfour( 13) =  13
          nfour( 14) =  13
          nfour( 15) =  14
          nfour( 16) =  15
          nfour( 17) =  15
          nfour( 18) =  15
          nfour( 19) =  15
          nfour( 20) =  16
          nfour( 21) =  15
          nfour( 22) =  15
          nfour( 23) =  18
          nfour( 24) =  16
          nfour( 25) =  16
          nfour( 26) =  14
          nfour( 27) =  16
          nfour( 28) =  16
          nfour( 29) =  15
          nfour( 30) =   8
          nfour( 31) =  15
          nfour( 32) =  14
          nfour( 33) =   1
        endif
        if(idomain.eq. 11) then
          nfour(  1) =   4
          nfour(  2) =   7
          nfour(  3) =   9
          nfour(  4) =  10
          nfour(  5) =  11
          nfour(  6) =  12
          nfour(  7) =  13
          nfour(  8) =  14
          nfour(  9) =  15
          nfour( 10) =  15
          nfour( 11) =  16
          nfour( 12) =  17
          nfour( 13) =  17
          nfour( 14) =  18
          nfour( 15) =  19
          nfour( 16) =  19
          nfour( 17) =  20
          nfour( 18) =  20
          nfour( 19) =  21
          nfour( 20) =  21
          nfour( 21) =  21
          nfour( 22) =  22
          nfour( 23) =  22
          nfour( 24) =  22
          nfour( 25) =  22
          nfour( 26) =  22
          nfour( 27) =  22
          nfour( 28) =  22
          nfour( 29) =  21
          nfour( 30) =  21
          nfour( 31) =  21
          nfour( 32) =  20
          nfour( 33) =  20
          nfour( 34) =  20
          nfour( 35) =  20
          nfour( 36) =  20
          nfour( 37) =  19
          nfour( 38) =  18
          nfour( 39) =  16
          nfour( 40) =  14
          nfour( 41) =   1
          nfour( 42) =   1
          nfour( 43) =  19
          nfour( 44) =   1
        endif
        if(idomain.eq.139) then
          nfour(  1) =   3
          nfour(  2) =   5
          nfour(  3) =   6
          nfour(  4) =   8
          nfour(  5) =   9
          nfour(  6) =  10
          nfour(  7) =  10
          nfour(  8) =  11
          nfour(  9) =  11
          nfour( 10) =  12
          nfour( 11) =  13
          nfour( 12) =  13
          nfour( 13) =  13
          nfour( 14) =  13
          nfour( 15) =  14
          nfour( 16) =  15
          nfour( 17) =  15
          nfour( 18) =  15
          nfour( 19) =  15
          nfour( 20) =  15
          nfour( 21) =  16
          nfour( 22) =  16
          nfour( 23) =  15
          nfour( 24) =  15
          nfour( 25) =  16
          nfour( 26) =  15
          nfour( 27) =  15
          nfour( 28) =  13
          nfour( 29) =   8
          nfour( 30) =  15
          nfour( 31) =   1
          nfour( 32) =   1
        endif
        if(idomain.eq.172) then
          nfour(  1) =   4
          nfour(  2) =   6
          nfour(  3) =   8
          nfour(  4) =   9
          nfour(  5) =  10
          nfour(  6) =  11
          nfour(  7) =  11
          nfour(  8) =  12
          nfour(  9) =  13
          nfour( 10) =  13
          nfour( 11) =  14
          nfour( 12) =  15
          nfour( 13) =  15
          nfour( 14) =  16
          nfour( 15) =  17
          nfour( 16) =  17
          nfour( 17) =  17
          nfour( 18) =  17
          nfour( 19) =  18
          nfour( 20) =  18
          nfour( 21) =  18
          nfour( 22) =  19
          nfour( 23) =  19
          nfour( 24) =  19
          nfour( 25) =  20
          nfour( 26) =  20
          nfour( 27) =  19
          nfour( 28) =  20
          nfour( 29) =  20
          nfour( 30) =  19
          nfour( 31) =  20
          nfour( 32) =  20
          nfour( 33) =  19
          nfour( 34) =  20
          nfour( 35) =  19
          nfour( 36) =  19
          nfour( 37) =  19
          nfour( 38) =  19
          nfour( 39) =  19
          nfour( 40) =  18
          nfour( 41) =  18
          nfour( 42) =  18
          nfour( 43) =  17
          nfour( 44) =  10
          nfour( 45) =  15
          nfour( 46) =  17
          nfour( 47) =  11
          nfour( 48) =   4
          nfour( 49) =   1
          nfour( 50) =   1
          nfour( 51) =   1
          nfour( 52) =   1
        endif
        if(idomain.eq.236) then
          nfour(  1) =   3
          nfour(  2) =   4
          nfour(  3) =   7
          nfour(  4) =   8
          nfour(  5) =   9
          nfour(  6) =   9
          nfour(  7) =  10
          nfour(  8) =  10
          nfour(  9) =  12
          nfour( 10) =  13
          nfour( 11) =  12
          nfour( 12) =  12
          nfour( 13) =  13
          nfour( 14) =  13
          nfour( 15) =  13
          nfour( 16) =  13
          nfour( 17) =  15
          nfour( 18) =  13
          nfour( 19) =  13
          nfour( 20) =  12
          nfour( 21) =  13
          nfour( 22) =  13
          nfour( 23) =   8
          nfour( 24) =   1
          nfour( 25) =   1
          nfour( 26) =   1
        endif
