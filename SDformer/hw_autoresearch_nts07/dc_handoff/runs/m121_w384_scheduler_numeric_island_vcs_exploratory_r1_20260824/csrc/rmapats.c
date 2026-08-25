// file = 0; split type = patterns; threshold = 100000; total count = 0.
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include "rmapats.h"

void  schedNewEvent (struct dummyq_struct * I1437, EBLK  * I1432, U  I628);
void  schedNewEvent (struct dummyq_struct * I1437, EBLK  * I1432, U  I628)
{
    U  I1717;
    U  I1718;
    U  I1719;
    struct futq * I1720;
    struct dummyq_struct * pQ = I1437;
    I1717 = ((U )vcs_clocks) + I628;
    I1719 = I1717 & ((1 << fHashTableSize) - 1);
    I1432->I671 = (EBLK  *)(-1);
    I1432->I672 = I1717;
    if (0 && rmaProfEvtProp) {
        vcs_simpSetEBlkEvtID(I1432);
    }
    if (I1717 < (U )vcs_clocks) {
        I1718 = ((U  *)&vcs_clocks)[1];
        sched_millenium(pQ, I1432, I1718 + 1, I1717);
    }
    else if ((peblkFutQ1Head != ((void *)0)) && (I628 == 1)) {
        I1432->I674 = (struct eblk *)peblkFutQ1Tail;
        peblkFutQ1Tail->I671 = I1432;
        peblkFutQ1Tail = I1432;
    }
    else if ((I1720 = pQ->I1339[I1719].I694)) {
        I1432->I674 = (struct eblk *)I1720->I692;
        I1720->I692->I671 = (RP )I1432;
        I1720->I692 = (RmaEblk  *)I1432;
    }
    else {
        sched_hsopt(pQ, I1432, I1717);
    }
}
#ifdef __cplusplus
extern "C" {
#endif
void SinitHsimPats(void);
#ifdef __cplusplus
}
#endif
