import numpy as np
from astropy import units, constants
import astropy.io.ascii
import scipy.interpolate

def planck_intensity(xarr, temperature, uout='mks'):
    '''
    This function calculates the thermal emission intensity spectrum of a surface.

        Inputs:
            wavelength = numpy array of wavelengths (with astropy units) OR
                numpy array of frequencies
            temperature = a single number, the temperature (with astropy units)

        Outputs:
            Returns an array of thermal emission intensities,
            in astropy units of W/(m^2*micron*sr). This is a flux, which has
            already been integrated over solid angle.
    '''

    # define variables as shortcut to the constants we need
    h = constants.h
    k = constants.k_B
    c = constants.c

    # this is the thing that goes into the exponent (its units better cancel!)
    if xarr.decompose().unit=='1/s':
        u = (h*xarr/(k*temperature)).decompose()
        intensity = (2*h*xarr**3/c**2/(np.exp(u) - 1))/units.steradian
        uout = 'cgs'
    else:
        u = h*c/(xarr*k*temperature)
        # calculate the intensity from the Planck function
        intensity = (2*h*c**2/xarr**5/(np.exp(u) - 1))/units.steradian
        uout = 'mks'

    # this isn't sufficiently general, need to be able to use MKS or CGS
    # with either per wavelength or per frequency, forcing behavior above
    # return the intensity
    if uout=='mks':
        return intensity.to('W/(m**2*micron*sr)')
    else: # cgs
        return intensity.to('erg/(cm**2*s*Hz*sr)')



def planck_flux(wavelength, temperature):
    '''
    This function calculates the thermal emission flux spectrum of a surface.

        Inputs:
            wavelength = numpy array of wavelengths (with astropy units)
            temperature = a single number, the temperature (with astropy units)

        Outputs:
            Returns an array of thermal emission fluxes,
            in astropy units of W/(m^2*micron). This is a flux, which has
            already been integrated over solid angle.
    '''

    # calculate the flux, knowing the angle integral will be pi steradians (for isotropic emission)
    flux = planck_intensity(wavelength, temperature)*np.pi*units.steradian

    # return the flux, in convenient units
    return flux.to('W/(m**2*micron)')


def xyz2rgb(X, Y, Z):
    '''
    This function converts CIE XYZ values into CIE RGB values.
    '''

    # normalize these, so they're all between 0 and 1
    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)
    z = Z/(X+Y+Z)

    # make a single column matrix containing the x,y,z values
    xyz = np.matrix([x,y,z]).T

    # rgb = conversion * xyz (with matrix math)
    conversion = np.matrix([[0.41847, -0.15866, -0.082835],
                            [-0.091169, 0.25243, 0.015708],
                            [0.00092090, -0.0025498, 0.17860]])

    # calculate the rgb single-column matrix
    color_matrix = conversion*xyz

    # convert to an array, and normalize so it doesn't exceed 1
    color = np.array(color_matrix.T)[0]
    color = color/np.max(color)

    return color



def spectrum2color(w, f):
    '''
    This function takes a spectrum.
    and returns its RGB color.

        w = wavelength (with astropy units attached)
        f = flux (with astropy units convertible to W/(nm m**2))

    '''
    
    # load the color matching functions as an astropy table
    cie = astropy.io.ascii.read('ciexyz31.csv')
    x = scipy.interpolate.interp1d(cie['wavelength'], cie['X'], fill_value=0.0, bounds_error=False)
    y = scipy.interpolate.interp1d(cie['wavelength'], cie['Y'], fill_value=0.0, bounds_error=False)
    z = scipy.interpolate.interp1d(cie['wavelength'], cie['Z'], fill_value=0.0, bounds_error=False)

    w_nm = w.to('nm').value

    # calculate the three integrals ("filter" response * spectrum)
    X = np.trapz(f*x(w_nm), w)
    Y = np.trapz(f*y(w_nm), w)
    Z = np.trapz(f*z(w_nm), w)

    # convert this XYZ into and RGB color
    color=xyz2rgb(X, Y, Z)

    return color
