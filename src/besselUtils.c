// various things stolen from GSL for creating matern functions
// https://github.com/ampl/gsl/blob/master/specfunc/bessel_Knu.c

#include <Rmath.h>
#include <math.h>
#include <stdlib.h>

#define GSL_DBL_EPSILON 2.2204460492503131e-16
#define GSL_SUCCESS 0
#define GSL_SQRT_DBL_MAX 1.3407807929942596e+154
#define GSL_EMAXITER 11

struct gsl_sf_result_struct {
  double val;
  double err;
};
typedef struct gsl_sf_result_struct gsl_sf_result;


struct gsl_sf_result_e10_struct {
  double val;
  double err;
  int    e10;
};
typedef struct gsl_sf_result_e10_struct gsl_sf_result_e10;

struct gsl_cheb_series_struct {

  double * c;   /* coefficients                */
  size_t order; /* order of expansion          */
  double a;     /* lower interval point        */
  double b;     /* upper interval point        */

  size_t order_sp;

  double * f;   /* function evaluated at chebyschev points  */
};
typedef struct gsl_cheb_series_struct gsl_cheb_series;



/* nu = (x+1)/4, -1<x<1, 1/(2nu)(1/Gamma[1-nu]-1/Gamma[1+nu]) */
static double g1_dat[14] = {
		-1.14516408366268311786898152867,
		0.00636085311347084238122955495,
		0.00186245193007206848934643657,
		0.000152833085873453507081227824,
		0.000017017464011802038795324732,
		-6.4597502923347254354668326451e-07,
		-5.1819848432519380894104312968e-08,
		4.5189092894858183051123180797e-10,
		3.2433227371020873043666259180e-11,
		6.8309434024947522875432400828e-13,
		2.8353502755172101513119628130e-14,
		-7.9883905769323592875638087541e-16,
		-3.3726677300771949833341213457e-17,
		-3.6586334809210520744054437104e-20
};

static gsl_cheb_series g1_cs = {
		g1_dat,
		13,
		-1, 1,
		7
};

/* nu = (x+1)/4, -1<x<1,  1/2 (1/Gamma[1-nu]+1/Gamma[1+nu]) */
static double g2_dat[15] =
{
		1.882645524949671835019616975350,
		-0.077490658396167518329547945212,
		-0.018256714847324929419579340950,
		0.0006338030209074895795923971731,
		0.0000762290543508729021194461175,
		-9.5501647561720443519853993526e-07,
		-8.8927268107886351912431512955e-08,
		-1.9521334772319613740511880132e-09,
		-9.4003052735885162111769579771e-11,
		4.6875133849532393179290879101e-12,
		2.2658535746925759582447545145e-13,
		-1.1725509698488015111878735251e-15,
		-7.0441338200245222530843155877e-17,
		-2.4377878310107693650659740228e-18,
		-7.5225243218253901727164675011e-20
};


static gsl_cheb_series g2_cs = {
		g2_dat,
		14,
		-1, 1,
		8
};


static inline int
cheb_eval_e(const gsl_cheb_series * cs,
		const double x,
		gsl_sf_result * result)
{
	int j;
	double d  = 0.0;
	double dd = 0.0;

	double y  = (2.0*x - cs->a - cs->b) / (cs->b - cs->a);
	double y2 = 2.0 * y;

	double e = 0.0;

	for(j = cs->order; j>=1; j--) {
		double temp = d;
		d = y2*d - dd + cs->c[j];
		e += fabs(y2*temp) + fabs(dd) + fabs(cs->c[j]);
		dd = temp;
	}

	{
		double temp = d;
		d = y*d - dd + 0.5 * cs->c[0];
		e += fabs(y*temp) + fabs(dd) + 0.5 * fabs(cs->c[0]);
	}

	result->val = d;
	result->err = GSL_DBL_EPSILON * e + fabs(cs->c[cs->order]);

	return GSL_SUCCESS;
}

void Rtemme_gamma(double *nu, double * g_1pnu, double * g_1mnu, double *g1, double *g2) {

	const double anu = fabs(*nu);    /* functions are even */
	const double x = 4.0*anu - 1.0;
	gsl_sf_result r_g1;
	gsl_sf_result r_g2;

	cheb_eval_e(&g1_cs, x, &r_g1);
	cheb_eval_e(&g2_cs, x, &r_g2);

	*g1 = r_g1.val;
	*g2 = r_g2.val;
	*g_1mnu = 1.0/(r_g2.val + *nu * r_g1.val);
	*g_1pnu = 1.0/(r_g2.val - *nu * r_g1.val);
}










