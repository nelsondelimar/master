# Master Codes


First of all: **Feel free to help us!**


### 1 Description
This is a public repository which was created to keep all source codes that will be developed during my master's project. Computational subroutines in Fortran and Python will be implemented and it will be posted on this repository. Currently, we are developing codes for magnetic anomalies for sources with regular geometry (sphere and prism). After we will work on a few codes for magnetic data processing. Some of those source codes are available in Blakely (1995).


### 2 Contributors

[Nelson Ribeiro Filho](http://lattes.cnpq.br/1419249921258591) - Master student in Geophysics

[Rodrigo Bijani](http://lattes.cnpq.br/2331435604103641) - Doctor in Geophysics


### Source codes 
* Solid sphere:
  * Magnetic induced field and its components (**Done!**);
  * Calculated total field anomaly (**Done!**);
  * Approximated total field anomaly (**Done!**);

* Rectangular prism:
  * Total field anomaly (**Done!**);

* Potential data filtering and transforms:
  * Upward and Downward continuation (**Done!**);
  * Horizontal and vertical derivatives (**Done!**);
  * Total gradient amplitude (**Done!**);
  * Reduction filter (**Done!**);
  * Tilt angle (**Done!**);
  
* Data analysis and processing:
  * Simple cross-correlation coefficient (**Done!**).

*  Classical equivalent layer technique:
  * Equivalent layer for horizontal and vertical derivatives (**Done!**);
  * Equivalent layer for reduction to Pole filter (**Done!**).

### Next step - What are we going to do?
The next step for this project has two different sides and approaches: the first one is try to find the magnetization direction for a 3D sourceby using the simple cross correlation approach. We have tested this source code in two dataset: (i) at the Serra do Cabral magnetic anomaly, the second largest magnetic anomaly in the world; and (ii) at the Cabo Frio magnetic anomaly, which envolves a pair of anomalies that are located at the cities of Cabo Frio and Arraial do Cabo. We found an excelent inclination-declination values for the Serra do Cabral magnetic anomaly, both pretty similar with previous publications. For the second area, we have not found true values, once there was not a maximum distribution zone in all intervals, both positive or negative. In that case, we assumed that the magnetic source has inclination and/or declination less than 15°, which bring us to the second approach, the classical equivalent layer technique, once the reduction to Pole filter shows numerical instabilities in this intervals.

#### Warning!
It had better be aware that all files contained within this repository are in constant development.

<p align="center">
  <img height="400" src="https://www.whiteheatdesign.co.uk/wp-content/uploads/working-on-it-large.jpg" />
</p>
