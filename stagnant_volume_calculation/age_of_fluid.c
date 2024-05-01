/* C library: Solving the moment of age equations to calculate stagnant volume */

#include "udf.h"

/*  User Defined Scalars : (3)*/
/*  0: RTD tracer */
/*  1: m_1 */
/*  2: m_2 */
/*  2: m_3 */
/*  2: m_4 */

real tau = 3.0 * 1; /* instant of tracer injection (3 times the cycle time) */
real eps = 1.0e-3;

enum
{
	tracer,
	washout,
	M1,
	M2,
	M3,
	M4,
	N_REQUIRED_UDS
};

DEFINE_SOURCE(m1_source, c, t, dS, eqn)
{
	real flow_time = RP_Get_Real("flow-time");

	dS[eqn] = 0.;
	if (flow_time > tau)
	{
		return C_R(c, t);
	}
	else
	{
		return 0.0;
	}
}

DEFINE_SOURCE(m2_source, c, t, dS, eqn)
{
	real flow_time = RP_Get_Real("flow-time");

	dS[eqn] = 0.;
	if (flow_time > tau)
	{
		return 2.0 * C_R(c, t) * C_UDSI(c, t, M1);
	}
	else
	{
		return 0.0;
	}
}

DEFINE_SOURCE(m3_source, c, t, dS, eqn)
{
	real flow_time = RP_Get_Real("flow-time");

	dS[eqn] = 0.;
	if (flow_time > tau)
	{
		return 3.0 * C_R(c, t) * C_UDSI(c, t, M2);
	}
	else
	{
		return 0.0;
	}
}

DEFINE_SOURCE(m4_source, c, t, dS, eqn)
{
	real flow_time = RP_Get_Real("flow-time");

	dS[eqn] = 0.;
	if (flow_time > tau)
	{
		return 4.0 * C_R(c, t) * C_UDSI(c, t, M3);
	}
	else
	{
		return 0.0;
	}
}

DEFINE_DIFFUSIVITY(uds_diff_coeff, c, t, i)
{
	real schmidt, d_turb, d_eff, mu_turb;
	d_eff = 1E-11; 
	d_eff *= C_R(c, t);

	if (rp_turb)
	{
		mu_turb = C_MU_T(c, t);
		schmidt = RP_Get_Real("species/sct");
		d_turb = mu_turb / schmidt;
		d_eff += d_turb;
	}
	return d_eff;
}

DEFINE_PROFILE(unsteady_tracer_profile, t, nv)
{
	face_t f;

	real flow_time = RP_Get_Real("flow-time");

	begin_f_loop(f, t)
	{
		if (flow_time > tau && flow_time < tau + 0.05)
		{
			F_PROFILE(f, t, nv) = 1.;
		}
		else
		{
			F_PROFILE(f, t, nv) = 0.;
		}
	}
	end_f_loop(f, t)
}

DEFINE_PROFILE(unsteady_washout_profile, t, nv)
{
	face_t f;

	real flow_time = RP_Get_Real("flow-time");

	begin_f_loop(f, t)
	{
		if (flow_time > tau)
		{
			F_PROFILE(f, t, nv) = 0.;
		}
		else
		{
			F_PROFILE(f, t, nv) = 1.;
		}
	}
	end_f_loop(f, t)
}

DEFINE_ADJUST(adjust_fcn, domain)
{
	Thread *t;
	cell_t c;
	real sigma2;
	real mu_prima_4;

	thread_loop_c(t, domain)
	{
		begin_c_loop(c, t)
		{
			/* CoV */
			if (C_UDSI(c, t, M1) > eps && C_UDSI(c, t, M2) > eps)
				C_UDMI(c, t, 0) = pow(C_UDSI(c, t, M2) / (C_UDSI(c, t, M1) * C_UDSI(c, t, M1)) - 1, 0.5);
			else
				C_UDMI(c, t, 0) = 0.;

			/* First normalized age moment */
			if (C_UDSI(c, t, M2) > eps && C_UDSI(c, t, M1) > eps)
			{
				sigma = pow(C_UDSI(c, t, M2) - C_UDSI(c, t, M1) * C_UDSI(c, t, M1), 0.5);
				mu_1 = C_UDSI(c, t, M1);
				C_UDMI(c, t, 1) = mu_1 / sigma;
			}
			else
				C_UDMI(c, t, 1) = 0.;
			
			/* Fourth normalized age moment */
			if (C_UDSI(c, t, M4) > eps && C_UDSI(c, t, M2) > eps && C_UDSI(c, t, M1) > eps)
			{
				sigma2 = C_UDSI(c, t, M2) - C_UDSI(c, t, M1) * C_UDSI(c, t, M1);
				mu_4 = C_UDSI(c, t, M4);
				C_UDMI(c, t, 2) = mu_4 / (sigma2 * sigma2);
			}
			else
				C_UDMI(c, t, 2) = 0.;
		}
		end_c_loop(c, t)
	}
}
