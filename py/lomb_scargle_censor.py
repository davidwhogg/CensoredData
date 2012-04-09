from numpy import empty,pi,sqrt,sin,cos,var,dot,array,int,clip
from scipy import weave
from scipy.optimize import fmin

def lomb(time, signal, error, time_missing, f1, df, numf, err_thresh=0.3, err_zpt=0.01):
    """
    C version of lomb_scargle

    Inputs:
        time: time vector
        signal: data vector
        error: data uncertainty vector
        time_missing: time vector for missing observations
        df: frequency step
        numf: number of frequencies to consider
        err_thresh: uncertainty of data not observed
        err_zpt: uncertainty in the zero-point, corresponding to uncertaintly in the limit prior

    Output:
        psd: power spectrum on frequency grid: f1,f1+df,...,f1+numf*df
    """
    numt = len(time)
    numt_missing = len(time_missing)

    wth = (1./error).astype('float64')
    s0 = dot(wth,wth)
    wth /= sqrt(s0)

    cn = (signal*wth).astype('float64')
    cn -= dot(cn,wth)*wth

    tt = 2*pi*time.astype('float64')
    sinx,cosx = sin(tt*f1)*wth,cos(tt*f1)*wth
    wpi = sin(df*tt); wpr=sin(0.5*df*tt); wpr = -2.*wpr*wpr

    tt = 2*pi*time_missing.astype('float64')
    sinx1,cosx1 = sin(tt*f1),cos(tt*f1)
    wpi1 = sin(df*tt); wpr1=sin(0.5*df*tt); wpr1 = -2.*wpr1*wpr1

    b0 = (cn/wth).max()
    thresh = array([b0,err_zpt,sqrt(err_thresh**2+err_zpt**2)],dtype='float64')

    # lookup tables for error function and it's derivatives
    #from scipy import derivative
    #from scipy.special import erf
    #f = lambda x: -2*log( 0.5 + 0.5*erf(x/sqrt(2)) )
    #x = arange(21.,dtype='float64')*0.5 - 5
    #df = derivative(f,x,n=1)
    #d2f = derivative(f,x,n=2)
    f = array([3.01299968e+01,2.51848395e+01,2.07202030e+01,1.67321306e+01,1.32154524e+01,1.01632966e+01,7.56636867e+00,5.41188880e+00,3.68204329e+00,2.35182352e+00,1.38629436e+00,7.37892831e-01,3.45507558e-01,1.38286911e-01,4.60258187e-02,1.24580510e-02,2.70161993e-03,4.65312283e-04,6.33434868e-05,6.79535779e-06,5.73303226e-07],dtype='float64')/s0
    df = array([-10.3766674,-9.41331104,-8.45727217,-7.51077146,-6.57691715,-5.66012091,-4.76670458,-3.90573652,-3.09003715,-2.33699799,-1.66826787,-1.10676831,-0.670134271,-0.362717390,-0.171402969,-0.0689107995,-0.0229812376,-0.00622562781,-0.00135052331,-0.000232637152,-0.0000316707568],dtype='float64')/s0
    d2f = array([1.93374719,1.92120438,1.9050432800,1.88387479,1.85566675,1.81742631,1.7647584,1.69134247,1.58857645,1.44613459,1.25496213,1.01432477,0.741305064,0.473777059,0.256157541,0.113836122,0.0406859223,0.0115342218,0.00257550626,0.000451759546,0.0000621988535],dtype='float64')/s0

    b = b0
    for i in xrange(5):
        x = 2*((b-cn/wth)/thresh[1]+5);
        jj = (x+0.5).astype('int32').clip(0,20);
        psd0 = -(f[jj]+0.5*(x-jj)*df[jj]).sum()
        fp = (df[jj]+0.5*(x-jj)*d2f[jj]).sum()/thresh[1]
        f2p = d2f[jj].sum()/thresh[1]**2
        x = 2*(-b/thresh[2]+5);
        jj = clip(int(x+0.5),0,20)
        psd0 -= (f[jj]+0.5*(x-jj)*df[jj])*numt_missing
        fp -= (df[jj]+0.5*(x-jj)*d2f[jj])*numt_missing/thresh[2]
        f2p += d2f[jj]*numt_missing/thresh[2]**2
        b -= fp/f2p

    thresh[0] = b
    lomb_code = """
      int i,l,jj;
      double b0=thresh[0], db=thresh[1], db1 = thresh[2], db2=db*db, db12=db1*db1;
      double mdetm, v0, v1, v2, v3, dp0, dp1, dp2, dp3;
      double d2p00, d2p01, d2p02, d2p03, d2p11, d2p12, d2p13, d2p22, d2p23, d2p33;
      double b, A0, A, B, m, dfj, d2fj, tmp, sum, x;
      for (unsigned long j=0;j<numf;j++) {

        // basic lomb-scargle
        //   m = A0 + A*sin + B*cos
        //
        double s1=0.,c1=0.,cs=0.,s2=0.,c2=0.,sh=0.,ch=0.,px=0;
        for (i=0;i<numt;i++) {
          s1 += sinx[i]*wth[i];
          c1 += cosx[i]*wth[i];
          cs += cosx[i]*sinx[i];
          c2 += cosx[i]*cosx[i];
          sh += sinx[i]*cn[i];
          ch += cosx[i]*cn[i];
        }
        s2 = 1-c2;

        // lomb-scargle upper limit part, w/ a Newton-Raphson update starting from basic LS
        //   initial guesses:
        b = b0; A0 = 0.; A = 0.; B=0.;
        dp1 = 0.; dp2 = -2*sh; dp3 = -2*ch;
        d2p11 = 2.; d2p12 = 2*s1; d2p13 = 2*c1;
        d2p22 = 2*s2; d2p23 = 2*cs; d2p33 = 2*c2;
        mdetm = d2p12*(d2p12*d2p33 - 2*d2p13*d2p23) + d2p13*d2p13*d2p22 - (d2p22*d2p33 - d2p23*d2p23)*d2p11;
        if (mdetm<0) {
          A0 = ((d2p23*dp3 - d2p33*dp2)*d2p12 - (d2p22*dp3 - d2p23*dp2)*d2p13 + (d2p22*d2p33 - d2p23*d2p23)*dp1)/mdetm;
          A = (d2p12*d2p13*dp3 - d2p13*d2p13*dp2 - (d2p23*dp3 - d2p33*dp2)*d2p11 - (d2p12*d2p33 - d2p13*d2p23)*dp1)/mdetm;
          B = (- d2p12*d2p12*dp3 + d2p12*d2p13*dp2 + (d2p22*dp3 - d2p23*dp2)*d2p11 + (d2p12*d2p23 - d2p13*d2p22)*dp1)/mdetm;
        }
        //printf(\"%f %f %f %f %f\\n\",b,A0,A,B,mdetm);

        dp0 = 0.;
        dp2 = 2*(-sh+A*s2+B*cs+A0*s1);
        dp3 = 2*(-ch+A*cs+B*c2+A0*c1);
        d2p00 = d2p02 = d2p03 = 0.;
        d2p22 = 2*s2; d2p23 = 2*cs; d2p33 = 2*c2;
        for (i=0;i<numt_missing;i++) {
          // f = -2*log( 0.5 + 0.5*erf( (m-b)/db1/sqrt(2) ) )
          m = A0 + A*sinx1[i] + B*cosx1[i];
          x = 2*((m-b)/db1+5);
          jj = (int)(x+0.5);
          if (jj<0) jj=0;
          else if (jj>20) jj=20;
          dfj = (df[jj]+0.5*(x-jj)*d2f[jj])/db1; d2fj = d2f[jj]/db12;
          dp0 -= dfj;
          dp2 += dfj*sinx1[i];
          dp3 += dfj*cosx1[i];
          d2p00 += d2fj;
          d2p02 -= (tmp=d2fj*sinx1[i]);
          d2p22 += tmp*sinx1[i];
          d2p23 += tmp*cosx1[i];
          d2p03 -= (tmp=d2fj*cosx1[i]);
          d2p33 += tmp*cosx1[i];
        }
        dp1 = 2.*(A*s1+B*c1+A0)-dp0; d2p11 = 2.+d2p00; d2p01 = -d2p00;
        d2p12 = 2*s1-d2p02; d2p13 = 2*c1-d2p03;
        for (i=0;i<numt;i++) {
          // f = -2*log( 0.5 + 0.5*erf( (b-cn[i]/wth[i])/db/sqrt(2) ) )
          x = 2*((b-cn[i]/wth[i])/db+5);
          jj = (int)(x+0.5);
          if (jj<0) jj=0;
          else if (jj>20) jj=20;
          dp0 += (df[jj]+0.5*(x-jj)*d2f[jj])/db; d2p00 += d2f[jj]/db2;
        }
        mdetm = (d2p22*d2p33 - d2p23*d2p23)*d2p01*d2p01 - 2*(d2p12*d2p33 - d2p13*d2p23)*d2p01*d2p02 + (d2p11*d2p33 - d2p13*d2p13)*d2p02*d2p02 + (d2p11*d2p22 - d2p12*d2p12)*d2p03*d2p03 + 2*((d2p12*d2p23 - d2p13*d2p22)*d2p01 - (d2p11*d2p23 - d2p12*d2p13)*d2p02)*d2p03 + (d2p12*d2p12*d2p33 - 2*d2p12*d2p13*d2p23 + d2p13*d2p13*d2p22 - (d2p22*d2p33 - d2p23*d2p23)*d2p11)*d2p00;
        if (mdetm<=0) {
          v0 = (((d2p23*dp3 - d2p33*dp2)*d2p12 - (d2p22*dp3 - d2p23*dp2)*d2p13 + (d2p22*d2p33 - d2p23*d2p23)*dp1)*d2p01 + (d2p12*d2p13*dp3 - d2p13*d2p13*dp2 - (d2p23*dp3 - d2p33*dp2)*d2p11 - (d2p12*d2p33 - d2p13*d2p23)*dp1)*d2p02 - (d2p12*d2p12*dp3 - d2p12*d2p13*dp2 - (d2p22*dp3 - d2p23*dp2)*d2p11 - (d2p12*d2p23 - d2p13*d2p22)*dp1)*d2p03 + (d2p12*d2p12*d2p33 - 2*d2p12*d2p13*d2p23 + d2p13*d2p13*d2p22 - (d2p22*d2p33 - d2p23*d2p23)*d2p11)*dp0);
          v1 = ((d2p23*dp3 - d2p33*dp2)*d2p01*d2p02 - (d2p13*dp3 - d2p33*dp1)*d2p02*d2p02 - (d2p12*dp2 - d2p22*dp1)*d2p03*d2p03 - ((d2p22*dp3 - d2p23*dp2)*d2p01 - (d2p12*dp3 + d2p13*dp2 - 2*d2p23*dp1)*d2p02)*d2p03 + ((d2p22*d2p33 - d2p23*d2p23)*d2p01 - (d2p12*d2p33 - d2p13*d2p23)*d2p02 + (d2p12*d2p23 - d2p13*d2p22)*d2p03)*dp0 - ((d2p23*dp3 - d2p33*dp2)*d2p12 - (d2p22*dp3 - d2p23*dp2)*d2p13 + (d2p22*d2p33 - d2p23*d2p23)*dp1)*d2p00);
          v2 = -((d2p23*dp3 - d2p33*dp2)*d2p01*d2p01 - (d2p13*dp3 - d2p33*dp1)*d2p01*d2p02 - (d2p11*dp2 - d2p12*dp1)*d2p03*d2p03 + ((d2p11*dp3 - d2p13*dp1)*d2p02 - (d2p12*dp3 - 2*d2p13*dp2 + d2p23*dp1)*d2p01)*d2p03 + ((d2p12*d2p33 - d2p13*d2p23)*d2p01 - (d2p11*d2p33 - d2p13*d2p13)*d2p02 + (d2p11*d2p23 - d2p12*d2p13)*d2p03)*dp0 + (d2p12*d2p13*dp3 - d2p13*d2p13*dp2 - (d2p23*dp3 - d2p33*dp2)*d2p11 - (d2p12*d2p33 - d2p13*d2p23)*dp1)*d2p00);
          v3 = ((d2p22*dp3 - d2p23*dp2)*d2p01*d2p01 + (d2p11*dp3 - d2p13*dp1)*d2p02*d2p02 - (2*d2p12*dp3 - d2p13*dp2 - d2p23*dp1)*d2p01*d2p02 + ((d2p12*dp2 - d2p22*dp1)*d2p01 - (d2p11*dp2 - d2p12*dp1)*d2p02)*d2p03 + ((d2p12*d2p23 - d2p13*d2p22)*d2p01 - (d2p11*d2p23 - d2p12*d2p13)*d2p02 + (d2p11*d2p22 - d2p12*d2p12)*d2p03)*dp0 + (d2p12*d2p12*dp3 - d2p12*d2p13*dp2 - (d2p22*dp3 - d2p23*dp2)*d2p11 - (d2p12*d2p23 - d2p13*d2p22)*dp1)*d2p00);
          b -= v0/mdetm; A0 -= v1/mdetm; A -= v2/mdetm; B -= v3/mdetm;
          //printf(\"%f %f %f %f %f\\n\",b,A0,A,B,mdetm);

          px = - A0*A0 - A*(A*s2 - 2*(sh - B*cs)) - B*(B*c2 - 2*ch) - 2*A0*(A*s1 + B*c1);
          for (i=0;i<numt_missing;i++) {
            m = A0 + A*sinx1[i] + B*cosx1[i];
            x = 2*((m-b)/db1+5);
            jj = (int)(x+0.5);
            if (jj<0) jj=0;
            else if (jj>20) jj=20;
            px -= f[jj]+0.5*(x-jj)*df[jj];
            sinx1[i] = (wpr1[i]*(tmp=sinx1[i]) + wpi1[i]*cosx1[i]) + sinx1[i];
            cosx1[i] = (wpr1[i]*cosx1[i] - wpi1[i]*tmp) + cosx1[i];
          }
          for (i=0;i<numt;i++) {
            x = 2*((b-cn[i]/wth[i])/db+5);
            jj = (int)(x+0.5);
            if (jj<0) jj=0;
            else if (jj>20) jj=20;
            px -= f[jj]+0.5*(x-jj)*df[jj];
            sinx[i] = (wpr[i]*(tmp=sinx[i]) + wpi[i]*cosx[i]) + sinx[i];
            cosx[i] = (wpr[i]*cosx[i] - wpi[i]*tmp) + cosx[i];
          }
        }
        psd[j] = px;
      }
    """

    psd = empty(numf,dtype='float64')
    weave.inline(lomb_code, ['cn','wth','numt','numt_missing','numf','psd','wpi','wpr','sinx','cosx','wpi1','wpr1','sinx1','cosx1','thresh','f','df','d2f'],\
      force=0)

    return 0.5/var(cn,ddof=1)*(psd-psd0)
